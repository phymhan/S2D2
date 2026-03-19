#!/usr/bin/env python3
"""
Evaluate Fast-dLLM v2 on GSM8K.

Adapted from `reference/eval_gsm8k_sdar.py`, but uses Fast-dLLM v2 generation
utilities the same way as `v2/example_v2.py` (generate / generate_ssd).
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_utils import generate, generate_ssd
from generate_policy_utils import generate_ssd_policy


SYSTEM_PROMPT = "Solve the following math problem concisely and clearly and put your final answer within \\boxed{}."

SYSTEM_PROMPT_COT = (
    "Solve the following math problem step by step. "
    "Show your reasoning clearly. "
    "Put your final answer within \\boxed{}."
)

FEW_SHOT_PROMPT = """Here are a few examples of how to solve math problems step by step.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. After planting, there are 21 trees. The number of trees planted is 21 - 15 = 6. So the final answer is \\boxed{6}.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah has 32 chocolates and her sister has 42, so together they have 32 + 42 = 74 chocolates. They eat 35 of them, leaving 74 - 35 = 39 chocolates. So the final answer is \\boxed{39}.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason starts with 20 lollipops and ends with 12, so he gave away 20 - 12 = 8 lollipops. So the final answer is \\boxed{8}.

Now solve a new problem in the same style.

Question: {question}
Answer:"""


def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ascii_sparkline(values, max_len=50):
    if values is None or len(values) == 0:
        return ""
    n = len(values)
    if n > max_len and max_len > 0:
        step = n / float(max_len)
        idxs = [min(int(i * step), n - 1) for i in range(max_len)]
        vals = [values[i] for i in idxs]
    else:
        vals = list(values)
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmax, vmin):
        norm = [0.0 for _ in vals]
    else:
        rng = vmax - vmin
        norm = [(v - vmin) / rng for v in vals]
    blocks = "▁▂▃▄▅▆▇█"
    try:
        chars = [blocks[min(int(x * (len(blocks) - 1)), len(blocks) - 1)] for x in norm]
        return "".join(chars)
    except Exception:
        shades = " .:-=+*#%@"
        chars = [shades[min(int(x * (len(shades) - 1)), len(shades) - 1)] for x in norm]
        return "".join(chars)


def extract_ground_truth(ans_text: str):
    """Extract numeric ground-truth answer from GSM8K annotation."""
    if not isinstance(ans_text, str):
        return None
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", ans_text)
    if m:
        try:
            return float(m[-1]) if "." in m[-1] else int(m[-1])
        except Exception:
            pass
    nums = re.findall(r"([-+]?\d+(?:\.\d+)?)", ans_text)
    if nums:
        try:
            return float(nums[-1]) if "." in nums[-1] else int(nums[-1])
        except Exception:
            return None
    return None


def extract_predicted_answer(text: str):
    """Extract final numeric answer from model output."""
    if not isinstance(text, str):
        return None
    boxed = re.findall(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", text)
    if boxed:
        try:
            return float(boxed[-1]) if "." in boxed[-1] else int(boxed[-1])
        except Exception:
            pass
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1]) if "." in m[-1] else int(m[-1])
        except Exception:
            pass
    nums = re.findall(r"([-+]?\d+(?:\.\d+)?)", text)
    if nums:
        try:
            return float(nums[-1]) if "." in nums[-1] else int(nums[-1])
        except Exception:
            return None
    return None


def build_user_prompt(question: str, use_few_shot: bool, cot: bool) -> str:
    if use_few_shot:
        return FEW_SHOT_PROMPT.replace("{question}", question)
    if cot:
        return f"Question: {question}\nAnswer: Let's think step by step."
    return f"Question: {question}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K with Fast-dLLM v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B", type=str, help="HF model name or local path")
    model_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model weights",
    )
    model_group.add_argument(
        "--device_map",
        type=str,
        default=None,
        help='Transformers device_map. If unset, uses "cuda:0" when available else "cpu".',
    )

    # Generation settings (Fast-dLLM v2)
    gen_group = parser.add_argument_group("Generation Settings (Fast-dLLM v2)")
    gen_group.add_argument("--prompt_length", type=int, default=4096, help="Maximum prompt length in tokens")
    gen_group.add_argument("--max_gen_toks", type=int, default=512, help="Maximum generation length in tokens")
    gen_group.add_argument("--block_size", type=int, default=32, help="Fast-dLLM block size")
    gen_group.add_argument("--small_block_size", type=int, default=8, help="Fast-dLLM small block size")
    gen_group.add_argument("--use_block_cache", type=str2bool, default=True, help="Enable block cache")
    gen_group.add_argument("--threshold", type=float, default=0.9, help="Confidence threshold for unmasking")
    gen_group.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    gen_group.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    gen_group.add_argument(
        "--generate_fn",
        type=str,
        default="fast",
        choices=["fast", "ssd", "ssd_policy"],
        help='Use Fast-dLLM "fast", "ssd", or "ssd_policy" decoding',
    )

    # SSD-only knobs (match v2/example_v2.py)
    gen_group.add_argument(
        "--use_ssd_cache",
        # action=argparse.BooleanOptionalAction,
        type=str2bool,
        default=False,
        help="(SSD only) Enable intra-block SSD cache patching (requires --use_block_cache).",
    )
    gen_group.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0, help="(SSD only) Tempering factor applied to q/p ratios.")
    gen_group.add_argument("--min_ssd_span_length", type=int, default=1, help="(SSD only) Minimum contiguous span length to verify.")
    gen_group.add_argument(
        "--allow_resample",
        # action=argparse.BooleanOptionalAction,
        type=str2bool,
        default=True,
        help="(SSD only) Residual resample at first rejected token.",
    )
    gen_group.add_argument("--cache_ver", type=str2bool, default=False, help="Enable cache verification.")
    gen_group.add_argument("--draft_ver", type=str2bool, default=False, help="Enable draft verification.")
    gen_group.add_argument(
        "--do_verify_policy",
        type=str,
        default="mask_span_length",
        choices=["mask_span_length", "score_threshold", "score_hysteresis", "contextual_bandit_ucb"],
        help="(SSD policy only) Policy for deciding whether to run verification.",
    )
    gen_group.add_argument(
        "--do_verify_score_threshold",
        type=float,
        default=0.0,
        help="(SSD policy only) Threshold for score_threshold policy.",
    )
    gen_group.add_argument(
        "--hysteresis_threshold_on",
        type=float,
        default=0.0,
        help="(SSD policy only) Turn-on threshold for score_hysteresis.",
    )
    gen_group.add_argument(
        "--hysteresis_threshold_off",
        type=float,
        default=-1.0,
        help="(SSD policy only) Turn-off threshold for score_hysteresis.",
    )
    gen_group.add_argument(
        "--token_acceptance_estimator",
        type=str,
        default="hard_margin_threshold",
        choices=["hard_margin_threshold", "soft_entropy_negexp", "soft_renyi_2_entropy"],
        help="(SSD policy only) Estimator used by score-based policies.",
    )
    gen_group.add_argument(
        "--ssd_confidence_margin_threshold",
        type=float,
        default=0.05,
        help="(SSD policy only) Margin threshold for hard_margin_threshold estimator.",
    )
    gen_group.add_argument(
        "--ssd_entropy_temperature",
        type=float,
        default=1.0,
        help="(SSD policy only) Temperature for soft_entropy_negexp estimator.",
    )
    gen_group.add_argument(
        "--ucb_beta",
        type=float,
        default=1.0,
        help="(SSD policy only) Exploration coefficient for contextual_bandit_ucb.",
    )
    gen_group.add_argument(
        "--ucb_span_length_bins",
        type=int,
        default=2,
        help="(SSD policy only) Number of span-length bins for UCB context.",
    )
    gen_group.add_argument(
        "--ucb_block_progress_bins",
        type=int,
        default=2,
        help="(SSD policy only) Number of block-progress bins for UCB context.",
    )
    gen_group.add_argument(
        "--ucb_entropy_bins",
        type=int,
        default=2,
        help="(SSD policy only) Number of entropy bins for UCB context.",
    )
    gen_group.add_argument(
        "--ucb_entropy_source",
        type=str,
        default="span",
        choices=["span", "masked"],
        help="(SSD policy only) Entropy source for UCB context features.",
    )
    gen_group.add_argument(
        "--do_verify_score_type",
        type=str,
        default="difference_dynamic",
        choices=["difference_dynamic", "difference_static"],
        help="(SSD policy only) Score function for score-based policies.",
    )
    gen_group.add_argument(
        "--score_penalty_coef",
        type=float,
        default=2.0,
        help="(SSD policy only) Penalty coefficient used by score function.",
    )
    # gen_group.add_argument("--return_decoding_order", type=str2bool, default=False, help="Return decoding order stats (debug).")

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--split", type=str, default="test", choices=["train", "test"])
    eval_group.add_argument("--sample_n", type=int, default=10, help="If >0, sample N examples deterministically")
    eval_group.add_argument("--few_shot", type=str2bool, default=False, help="Use 3-shot prompt examples")
    eval_group.add_argument("--cot", type=str2bool, default=False, help="Enable chain-of-thought prompting")
    eval_group.add_argument("--no_chat", type=str2bool, default=False, help="Do not use tokenizer chat template")
    eval_group.add_argument("--use_majority_vote", type=str2bool, default=False, help="Majority vote over multiple samples")
    eval_group.add_argument("--n_votes", type=int, default=5, help="Number of votes when majority voting")
    eval_group.add_argument("--verbose", type=str2bool, default=False, help="Print per-example generations and predictions")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_samples", type=str2bool, default=False, help="Save per-sample outputs and summary")
    log_group.add_argument("--output_path", type=str, default=None, help="Path to write results (json/jsonl)")
    log_group.add_argument("--summary_file", type=str, default=None, help="Append one JSONL record with config + final accuracy")
    log_group.add_argument("--config_str", type=str, default=None, help="Optional config string recorded in summary")

    # Other settings
    other_group = parser.add_argument_group("Other Settings")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed")
    other_group.add_argument(
        "--use_local_modeling",
        type=str2bool,
        default=True,
        help="Load model class from local v2/modeling_fast.py instead of HF remote code.",
    )

    args = parser.parse_args()

    if args.max_gen_toks <= 0:
        parser.error("--max_gen_toks must be positive")
    if args.block_size <= 0:
        parser.error("--block_size must be positive")
    if args.small_block_size <= 0:
        parser.error("--small_block_size must be positive")

    if args.generate_fn == "ssd" and args.use_majority_vote:
        # SSD implementation supports batch_size=1, which we already use per-example.
        # Majority vote is still supported by running multiple independent generations.
        pass

    set_seed(args.seed)

    device_map = args.device_map
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"

    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Generate function: {args.generate_fn}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Device map: {device_map}")
    print(f"  Prompt length: {args.prompt_length}")
    print(f"  Max gen toks: {args.max_gen_toks}")
    print(f"  Block size: {args.block_size}")
    print(f"  Small block size: {args.small_block_size}")
    print(f"  Use block cache: {args.use_block_cache}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Few-shot: {args.few_shot}")
    print(f"  CoT: {args.cot}")
    print(f"  Majority vote: {args.use_majority_vote} (n={args.n_votes})")
    print(f"  Seed: {args.seed}")

    if args.generate_fn == "ssd":
        generate_fn = generate_ssd
    elif args.generate_fn == "ssd_policy":
        generate_fn = generate_ssd_policy
    else:
        generate_fn = generate

    # Mirror `reference/eval_gsm8k_sdar.py`: prepare kwargs once, outside the eval loop.
    generate_fn_kwargs = dict(
        max_gen_toks=args.max_gen_toks,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        use_block_cache=args.use_block_cache,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        return_decoding_order=False,
    )
    if args.generate_fn == "ssd":
        generate_fn_kwargs.update(
            dict(
                use_ssd_cache=args.use_ssd_cache,
                ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
                min_ssd_span_length=args.min_ssd_span_length,
                allow_resample=args.allow_resample,
                cache_ver=args.cache_ver,
                draft_ver=args.draft_ver,
            )
        )
    elif args.generate_fn == "ssd_policy":
        generate_fn_kwargs.update(
            dict(
                use_ssd_cache=args.use_ssd_cache,
                ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
                min_ssd_span_length=args.min_ssd_span_length,
                allow_resample=args.allow_resample,
                cache_ver=args.cache_ver,
                draft_ver=args.draft_ver,
                do_verify_policy=args.do_verify_policy,
                do_verify_score_threshold=args.do_verify_score_threshold,
                hysteresis_threshold_on=args.hysteresis_threshold_on,
                hysteresis_threshold_off=args.hysteresis_threshold_off,
                token_acceptance_estimator=args.token_acceptance_estimator,
                ssd_confidence_margin_threshold=args.ssd_confidence_margin_threshold,
                ssd_entropy_temperature=args.ssd_entropy_temperature,
                ucb_beta=args.ucb_beta,
                ucb_span_length_bins=args.ucb_span_length_bins,
                ucb_block_progress_bins=args.ucb_block_progress_bins,
                ucb_entropy_bins=args.ucb_entropy_bins,
                ucb_entropy_source=args.ucb_entropy_source,
                do_verify_score_type=args.do_verify_score_type,
                score_penalty_coef=args.score_penalty_coef,
            )
        )

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_local_modeling:
        from modeling_fast import Fast_dLLM_QwenForCausalLM
        model_cls = Fast_dLLM_QwenForCausalLM
        print("  Using local modeling file: modeling_fast.py")
    else:
        model_cls = AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)

    if args.sample_n is not None and args.sample_n > 0 and args.sample_n < len(dataset):
        rng = random.Random(123456)  # fixed for repeatability across runs
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        select = indices[: args.sample_n]
        select.sort()
        print(f"Selected indices (n={len(select)}): {select}")
        dataset = dataset.select(select)

    datasize = len(dataset)
    print(f"gsm8k {args.split} size:", datasize)

    results = []
    correct_so_far = 0
    acc_series = []

    mask_token_str = tokenizer.mask_token or "<|MASK|>"

    eval_t0 = time.perf_counter()
    pbar = tqdm(range(datasize), desc="Evaluating")
    for idx in pbar:
        ex = dataset[idx]
        question = ex["question"]
        gold_text = ex["answer"]
        gt = extract_ground_truth(gold_text)

        user_content = build_user_prompt(question, use_few_shot=args.few_shot, cot=args.cot)
        system_prompt = SYSTEM_PROMPT_COT if args.cot else SYSTEM_PROMPT
        prompt_text = f"{system_prompt}\n\n{user_content}"

        if args.no_chat:
            formatted = prompt_text
        else:
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        tokenize_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=args.prompt_length,
        )
        tokens = tokenizer.batch_encode_plus([formatted], **tokenize_kwargs)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        prompt_len = int(tokens["input_ids"].shape[1])

        batch_model_answers = []
        n_votes = args.n_votes if args.use_majority_vote else 1
        example_seed_base = args.seed + idx * 100000

        for vote_i in range(n_votes):
            set_seed(example_seed_base + vote_i)

            out = generate_fn(
                model,
                tokens["input_ids"],
                tokenizer=tokenizer,
                **generate_fn_kwargs,
            )

            stats = None
            if isinstance(out, tuple):
                out, stats = out

            seq = out.sequences if hasattr(out, "sequences") else out
            gen_ids = seq[:, prompt_len : prompt_len + args.max_gen_toks]
            out_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            out_text = out_text.replace(mask_token_str, "").replace("<|MASK|>", "")
            pred_num = extract_predicted_answer(out_text)

            batch_model_answers.append({"text": out_text, "numeric": pred_num, "stats": stats})

        numeric_answers = [ma["numeric"] for ma in batch_model_answers]
        filtered = [num for num in numeric_answers if num is not None]
        majority = Counter(filtered).most_common(1)[0][0] if filtered else None

        correct = (majority == gt) if majority is not None and gt is not None else False
        if correct:
            correct_so_far += 1

        results.append(
            {
                "question": question,
                "gold_answer_text": gold_text,
                "model_answers_text": [ma["text"] for ma in batch_model_answers],
                "extracted_model_answers": numeric_answers,
                "extracted_gold_answer": gt,
                "majority_answer": majority,
                "correct": correct,
            }
        )

        if args.verbose:
            print(f"\n[Example {idx}]")
            print(f"Question: {question}")
            print(f"Gold answer text: {gold_text}")
            print(f"Model answer text: {batch_model_answers[0]['text']}")
            print(f"Predictions: {numeric_answers}")
            print(f"Majority: {majority}, GT: {gt}, Correct: {correct}")

        processed = len(results)
        acc_so_far = correct_so_far / float(processed)
        acc_series.append(acc_so_far)
        pbar.set_postfix({"acc": f"{acc_so_far:.4f}", "trend": ascii_sparkline(acc_series, max_len=80)})

    eval_seconds = time.perf_counter() - eval_t0

    cnt = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = cnt / total if total > 0 else 0.0
    print(f"Accuracy: {cnt} / {total} = {acc :.4f}")

    results.append({"accuracy": acc})

    if args.summary_file:
        summary_base = Path(args.summary_file)
        if summary_base.suffix in {".json", ".jsonl", ".txt"}:
            summary_base = summary_base.with_suffix("")

        summary_jsonl_path = summary_base.with_suffix(".jsonl")
        summary_txt_path = summary_base.with_suffix(".txt")

        if summary_base.parent and str(summary_base.parent) != ".":
            summary_base.parent.mkdir(parents=True, exist_ok=True)

        summary_rec = {
            "config": args.config_str,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "acc": acc,
            "correct": cnt,
            "total": total,
            "eval_seconds": eval_seconds,
            "max_gen_toks": args.max_gen_toks,
            "block_size": args.block_size,
            "small_block_size": args.small_block_size,
            "threshold": args.threshold,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "generate_fn": args.generate_fn,
            "use_ssd_cache": bool(args.use_ssd_cache),
            "ssd_ratio_tempering_factor": args.ssd_ratio_tempering_factor,
            "min_ssd_span_length": args.min_ssd_span_length,
            "allow_resample": bool(args.allow_resample),
            "do_verify_policy": args.do_verify_policy,
            "do_verify_score_threshold": args.do_verify_score_threshold,
            "hysteresis_threshold_on": args.hysteresis_threshold_on,
            "hysteresis_threshold_off": args.hysteresis_threshold_off,
            "token_acceptance_estimator": args.token_acceptance_estimator,
            "ssd_confidence_margin_threshold": args.ssd_confidence_margin_threshold,
            "ssd_entropy_temperature": args.ssd_entropy_temperature,
            "ucb_beta": args.ucb_beta,
            "ucb_span_length_bins": args.ucb_span_length_bins,
            "ucb_block_progress_bins": args.ucb_block_progress_bins,
            "ucb_entropy_bins": args.ucb_entropy_bins,
            "ucb_entropy_source": args.ucb_entropy_source,
            "do_verify_score_type": args.do_verify_score_type,
            "score_penalty_coef": args.score_penalty_coef,
            "use_block_cache": bool(args.use_block_cache),
            "model": args.model,
            "sample_n": args.sample_n,
            "seed": args.seed,
            "argv": list(sys.argv),
        }
        with open(summary_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

        txt_line = (
            f"config={summary_rec['config']} "
            f"acc={acc:.4f} ({cnt}/{total}) "
            f"eval_seconds={eval_seconds:.3f} "
            f"max_gen_toks={args.max_gen_toks} "
            f"block_size={args.block_size} "
            f"small_block_size={args.small_block_size} "
            f"threshold={args.threshold} "
            f"top_p={args.top_p} "
            f"temperature={args.temperature} "
            f"generate_fn={args.generate_fn} "
            f"use_block_cache={int(bool(args.use_block_cache))} "
            f"use_ssd_cache={int(bool(args.use_ssd_cache))} "
            f"min_ssd_span_length={args.min_ssd_span_length} "
            f"ssd_ratio_tempering_factor={args.ssd_ratio_tempering_factor} "
            f"allow_resample={int(bool(args.allow_resample))} "
            f"do_verify_policy={args.do_verify_policy} "
            f"do_verify_score_threshold={args.do_verify_score_threshold} "
            f"hysteresis_threshold_on={args.hysteresis_threshold_on} "
            f"hysteresis_threshold_off={args.hysteresis_threshold_off} "
            f"token_acceptance_estimator={args.token_acceptance_estimator} "
            f"ssd_confidence_margin_threshold={args.ssd_confidence_margin_threshold} "
            f"ssd_entropy_temperature={args.ssd_entropy_temperature} "
            f"ucb_beta={args.ucb_beta} "
            f"ucb_span_length_bins={args.ucb_span_length_bins} "
            f"ucb_block_progress_bins={args.ucb_block_progress_bins} "
            f"ucb_entropy_bins={args.ucb_entropy_bins} "
            f"ucb_entropy_source={args.ucb_entropy_source} "
            f"do_verify_score_type={args.do_verify_score_type} "
            f"score_penalty_coef={args.score_penalty_coef} "
            f"model={args.model} "
            f"sample_n={args.sample_n} "
            f"seed={args.seed}"
        )
        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write(txt_line + "\n")

    if args.log_samples and args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ext = os.path.splitext(args.output_path)[1].lower()
        if ext == ".jsonl":
            with open(args.output_path, "w", encoding="utf-8") as f:
                for rec in results[:-1]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
        else:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {args.output_path}")

    if len(acc_series) > 1:
        print("Accuracy trend:")
        print(ascii_sparkline(acc_series, max_len=80))


if __name__ == "__main__":
    main()

