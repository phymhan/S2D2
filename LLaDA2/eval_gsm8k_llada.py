#!/usr/bin/env python3
"""
Evaluate an LLaDA2.1 (KV-cache) model on GSM8K.

Uses generate_cached() / generate_ssd_policy() from generate_utils.py.
Adapted from references/eval_gsm8k_sdar.py.
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

from generate_utils import generate, generate_cached, generate_ssd_policy, load_model_and_tokenizer
from utils import set_seed, str2bool

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K with LLaDA2.1 (KV-cache)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model Configuration ──
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default="inclusionAI/LLaDA2.1-mini", help="HuggingFace model repo or local path")
    model_group.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype for model weights")
    model_group.add_argument("--device_map", type=str, default="auto", help="Transformers device_map value")
    model_group.add_argument("--cuda_visible_devices", "--cvd", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES before loading the model")
    model_group.add_argument("--mask_id", type=int, default=156895, help="Mask token id for diffusion")
    model_group.add_argument("--eos_id", type=int, default=156892, help="EOS token id for early stopping")

    # ── Generation Settings ──
    gen_group = parser.add_argument_group("Generation Settings")
    gen_group.add_argument("--gen_length", type=int, default=256, help="Maximum generation length in tokens")
    gen_group.add_argument("--block_length", type=int, default=32, help="Block length for diffusion decoding")
    gen_group.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    gen_group.add_argument("--top_k", type=int, default=None, help="Top-k sampling cutoff")
    gen_group.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold")
    gen_group.add_argument("--threshold", type=float, default=0.7, help="Acceptance threshold for unmasking tokens")
    gen_group.add_argument("--editing_threshold", type=float, default=0.5, help="Editing threshold for generation")
    gen_group.add_argument("--max_post_steps", type=int, default=0, help="Post-mask global editing steps per block")
    gen_group.add_argument("--num_to_transfer", type=int, default=1, help="Minimum masked positions to resolve per iteration")
    gen_group.add_argument("--eos_early_stop", type=str2bool, default=True, help="Enable/disable early stopping at EOS")
    gen_group.add_argument("--generate_fn", type=str, default="cached", choices=["nocache", "cached", "ssd_policy"], help="Generation function to use")

    # ── SSD-specific ──
    ssd_group = parser.add_argument_group("SSD-specific (ssd_policy)")
    ssd_group.add_argument("--min_ssd_span_length", type=int, default=1, help="Minimum mask span length to trigger 2L verification")
    ssd_group.add_argument("--legacy_ssd_span_strategy", type=str2bool, default=False, help="If set, mask_span_length policy also checks high-confidence count")
    ssd_group.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0, help="Exponent applied to SSD acceptance ratios")
    ssd_group.add_argument("--do_verify_policy", type=str, default="mask_span_length", choices=["mask_span_length", "score_threshold", "score_hysteresis"], help="Policy for deciding whether to run the 2L verifier")
    ssd_group.add_argument("--do_verify_score_threshold", type=float, default=0.0, help="Threshold for score_threshold policy")
    ssd_group.add_argument("--hysteresis_threshold_on", type=float, default=0.0, help="Turn-on threshold for score_hysteresis policy")
    ssd_group.add_argument("--hysteresis_threshold_off", type=float, default=-1.0, help="Turn-off threshold for score_hysteresis policy")
    ssd_group.add_argument("--do_verify_score_type", type=str, default="difference_dynamic", choices=["difference_dynamic", "difference_static"], help="Score function for score-based verify policies")
    ssd_group.add_argument("--score_penalty_coef", type=float, default=2.0, help="Penalty coefficient c in score computation")
    ssd_group.add_argument("--token_acceptance_estimator", type=str, default="hard_margin_threshold", choices=["hard_margin_threshold", "soft_entropy_negexp", "soft_renyi_2_entropy"], help="Estimator for per-token acceptance probability")
    ssd_group.add_argument("--ssd_confidence_margin_threshold", type=float, default=0.05, help="Margin threshold for hard_margin_threshold estimator")
    ssd_group.add_argument("--ssd_entropy_temperature", type=float, default=1.0, help="Temperature for soft_entropy_negexp estimator")

    # ── Evaluation Settings ──
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--sample_n", type=int, default=10, help="Sample N test examples (0 or negative for full dataset)")
    eval_group.add_argument("--few_shot", type=str2bool, default=False, help="Use 3-shot prompt examples")
    eval_group.add_argument("--cot", type=str2bool, default=False, help="Enable chain-of-thought prompting")
    eval_group.add_argument("--no_chat", type=str2bool, default=False, help="Do not use tokenizer chat template")
    eval_group.add_argument("--use_majority_vote", type=str2bool, default=False, help="Majority vote over multiple samples")
    eval_group.add_argument("--n_votes", type=int, default=5, help="Number of votes when majority voting")
    eval_group.add_argument("--verbose", type=str2bool, default=False, help="Print per-example generations and predictions")
    eval_group.add_argument("--prompt_length", type=int, default=4096, help="Maximum prompt length in tokens")

    # ── Logging ──
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_samples", type=str2bool, default=False, help="Save per-sample outputs and summary")
    log_group.add_argument("--output_path", type=str, default=None, help="Path to write results (json/jsonl)")
    log_group.add_argument("--summary_file", type=str, default=None, help="Append one JSONL record with config + final accuracy")
    log_group.add_argument("--config_str", type=str, default=None, help="Tag string appended to the summary record")

    # ── Other ──
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    set_seed(args.seed)

    # ── Print configuration ──
    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Generate function: {args.generate_fn}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Device map: {args.device_map}")
    print(f"  Mask id: {args.mask_id}")
    print(f"  EOS id: {args.eos_id}")
    print(f"  Gen length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Editing threshold: {args.editing_threshold}")
    print(f"  Max post steps: {args.max_post_steps}")
    print(f"  Num to transfer: {args.num_to_transfer}")
    print(f"  Few-shot: {args.few_shot}")
    print(f"  CoT: {args.cot}")
    print(f"  Majority vote: {args.use_majority_vote} (n={args.n_votes})")
    print(f"  Seed: {args.seed}")

    # ── Load model ──
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype_str=args.dtype,
        device_map=args.device_map,
    )

    # ── Load dataset ──
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if args.sample_n is not None and args.sample_n > 0 and args.sample_n < len(dataset):
        rng = random.Random(123456)  # fixed for repeatability across runs
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        select = indices[: args.sample_n]
        select.sort()
        print(f"Selected indices (n={len(select)}): {select}")
        dataset = dataset.select(select)

    datasize = len(dataset)
    print("gsm8k test size:", datasize)

    # ── Set up generate function and static kwargs (like references/eval_gsm8k_sdar.py) ──
    gen_fn_kwargs = dict(
        temperature=args.temperature,
        block_length=args.block_length,
        gen_length=args.gen_length,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_early_stop=args.eos_early_stop,
        eos_id=args.eos_id,
        mask_id=args.mask_id,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        num_to_transfer=args.num_to_transfer,
    )

    if args.generate_fn == "nocache":
        gen_fn = generate
    elif args.generate_fn == "cached":
        gen_fn = generate_cached
    elif args.generate_fn == "ssd_policy":
        gen_fn = generate_ssd_policy
        gen_fn_kwargs.update(
            min_ssd_span_length=args.min_ssd_span_length,
            legacy_ssd_span_strategy=args.legacy_ssd_span_strategy,
            ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
            do_verify_policy=args.do_verify_policy,
            do_verify_score_threshold=args.do_verify_score_threshold,
            hysteresis_threshold_on=args.hysteresis_threshold_on,
            hysteresis_threshold_off=args.hysteresis_threshold_off,
            do_verify_score_type=args.do_verify_score_type,
            score_penalty_coef=args.score_penalty_coef,
            token_acceptance_estimator=args.token_acceptance_estimator,
            ssd_confidence_margin_threshold=args.ssd_confidence_margin_threshold,
            ssd_entropy_temperature=args.ssd_entropy_temperature,
        )
    else:
        raise ValueError(f"Unknown generate function: {args.generate_fn}")

    results = []
    correct_so_far = 0
    total_nfe = 0
    acc_series = []

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
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                formatted = prompt_text

        input_ids = tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=args.prompt_length,
        )["input_ids"].to(model.device)

        batch_model_answers = []
        n_votes = args.n_votes if args.use_majority_vote else 1

        for vote_i in range(n_votes):
            if n_votes > 1:
                set_seed(args.seed + 1000 + vote_i)

            generated_tokens, stats = gen_fn(
                model=model, input_ids=input_ids, **gen_fn_kwargs
            )
            total_nfe += stats["nfe"]

            # generated_tokens includes prompt; slice it off for answer extraction
            out_text = tokenizer.decode(generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred_num = extract_predicted_answer(out_text)
            batch_model_answers.append({"text": out_text, "numeric": pred_num})

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
    avg_nfe = total_nfe / total if total > 0 else 0.0
    print(f"Accuracy: {cnt} / {total} = {acc:.4f}")
    print(f"Total NFE: {total_nfe}, Avg NFE: {avg_nfe:.1f}")

    results.append({"accuracy": acc, "total_nfe": total_nfe, "avg_nfe": avg_nfe})

    # ── Summary logging ──
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
            "total_nfe": total_nfe,
            "avg_nfe": avg_nfe,
            "eval_seconds": eval_seconds,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "editing_threshold": args.editing_threshold,
            "max_post_steps": args.max_post_steps,
            "num_to_transfer": args.num_to_transfer,
            "generate_fn": args.generate_fn,
            "model": args.model,
            "sample_n": args.sample_n,
            "seed": args.seed,
            "argv": list(sys.argv),
        }
        if args.generate_fn == "ssd_policy":
            summary_rec.update(
                {
                    "min_ssd_span_length": args.min_ssd_span_length,
                    "legacy_ssd_span_strategy": args.legacy_ssd_span_strategy,
                    "ssd_ratio_tempering_factor": args.ssd_ratio_tempering_factor,
                    "do_verify_policy": args.do_verify_policy,
                    "do_verify_score_threshold": args.do_verify_score_threshold,
                    "hysteresis_threshold_on": args.hysteresis_threshold_on,
                    "hysteresis_threshold_off": args.hysteresis_threshold_off,
                    "do_verify_score_type": args.do_verify_score_type,
                    "score_penalty_coef": args.score_penalty_coef,
                    "token_acceptance_estimator": args.token_acceptance_estimator,
                    "ssd_confidence_margin_threshold": args.ssd_confidence_margin_threshold,
                    "ssd_entropy_temperature": args.ssd_entropy_temperature,
                }
            )
        with open(summary_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

        txt_line = (
            f"config={summary_rec['config']} "
            f"acc={acc:.4f} ({cnt}/{total}) "
            f"total_nfe={total_nfe} avg_nfe={avg_nfe:.1f} "
            f"eval_seconds={eval_seconds:.3f} "
            f"gen_length={args.gen_length} "
            f"block_length={args.block_length} "
            f"temperature={args.temperature} "
            f"threshold={args.threshold} "
            f"editing_threshold={args.editing_threshold} "
            f"max_post_steps={args.max_post_steps} "
            f"num_to_transfer={args.num_to_transfer} "
            f"generate_fn={args.generate_fn} "
            f"model={args.model} "
            f"sample_n={args.sample_n} "
            f"seed={args.seed}"
        )
        if args.generate_fn == "ssd_policy":
            txt_line += (
                f" min_ssd_span_length={args.min_ssd_span_length}"
                f" legacy_ssd_span_strategy={int(bool(args.legacy_ssd_span_strategy))}"
                f" ssd_ratio_tempering_factor={args.ssd_ratio_tempering_factor}"
                f" do_verify_policy={args.do_verify_policy}"
                f" do_verify_score_threshold={args.do_verify_score_threshold}"
                f" hysteresis_threshold_on={args.hysteresis_threshold_on}"
                f" hysteresis_threshold_off={args.hysteresis_threshold_off}"
                f" do_verify_score_type={args.do_verify_score_type}"
                f" score_penalty_coef={args.score_penalty_coef}"
                f" token_acceptance_estimator={args.token_acceptance_estimator}"
                f" ssd_confidence_margin_threshold={args.ssd_confidence_margin_threshold}"
                f" ssd_entropy_temperature={args.ssd_entropy_temperature}"
            )
        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write(txt_line + "\n")

    # ── Per-sample logging ──
    if args.log_samples and args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ext = os.path.splitext(args.output_path)[1].lower()
        if ext == ".jsonl":
            with open(args.output_path, "w") as f:
                for rec in results[:-1]:
                    f.write(json.dumps(rec) + "\n")
                f.write(json.dumps(results[-1]) + "\n")
        else:
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_path}")

    if len(acc_series) > 1:
        print("Accuracy trend:")
        print(ascii_sparkline(acc_series, max_len=80))


if __name__ == "__main__":
    main()
