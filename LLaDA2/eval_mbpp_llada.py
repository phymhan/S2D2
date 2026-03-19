#!/usr/bin/env python3
"""
Evaluate an LLaDA2.1 (KV-cache) model on MBPP.

Uses generate_cached() / generate_ssd_policy() from generate_utils.py.
Modeled after eval_gsm8k_llada.py with flexible code extraction and
subprocess-based test execution.
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
import subprocess
import sys
import tempfile
import time
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

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write only the function code, no explanations."
)

# ---------------------------------------------------------------------------
# Few-shot examples (same 3 as lm-eval's list_fewshot_samples)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
        "code": (
            "```python\n"
            "def similar_elements(test_tup1, test_tup2):\n"
            "  res = tuple(set(test_tup1) & set(test_tup2))\n"
            "  return (res)\n"
            "```"
        ),
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "code": (
            "```python\n"
            "import math\n"
            "def is_not_prime(n):\n"
            "    result = False\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            result = True\n"
            "    return result\n"
            "```"
        ),
    },
    {
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
        "code": (
            "```python\n"
            "import heapq as hq\n"
            "def heap_queue_largest(nums,n):\n"
            "  largest_nums = hq.nlargest(n, nums)\n"
            "  return largest_nums\n"
            "```"
        ),
    },
]

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


def build_user_message(text: str, test_list: list) -> str:
    """Build user message for a single MBPP task."""
    tests_str = "\n".join(test_list[:3])
    return f"{text}\nYour code should pass these tests:\n{tests_str}"


def build_chat_messages(text: str, test_list: list, num_fewshot: int) -> list:
    """Build multi-turn chat messages with optional few-shot examples."""
    messages = []
    n = min(num_fewshot, len(FEW_SHOT_EXAMPLES))
    for ex in FEW_SHOT_EXAMPLES[:n]:
        messages.append({"role": "user", "content": build_user_message(ex["text"], ex["test_list"])})
        messages.append({"role": "assistant", "content": ex["code"]})
    messages.append({"role": "user", "content": build_user_message(text, test_list)})
    return messages


# ---------------------------------------------------------------------------
# Code extraction (flexible, try in order)
# ---------------------------------------------------------------------------


def extract_code(response: str) -> str:
    """
    Extract Python code from model response.

    Tries in order:
    1. ```python ... ``` markdown blocks -> take longest
    2. ``` ... ``` generic code blocks -> take longest
    3. [BEGIN] ... [DONE] markers
    4. Heuristic: contiguous lines starting with def/import/from/class + indented
    5. Fallback: entire response
    """
    if not isinstance(response, str) or not response.strip():
        return ""

    # 1. ```python ... ``` blocks
    py_blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if py_blocks:
        return max(py_blocks, key=len).strip()

    # 2. ``` ... ``` generic blocks
    gen_blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if gen_blocks:
        return max(gen_blocks, key=len).strip()

    # 3. [BEGIN] ... [DONE] markers
    begin_done = re.findall(r"\[BEGIN\]\s*\n(.*?)\[DONE\]", response, re.DOTALL)
    if begin_done:
        return max(begin_done, key=len).strip()

    # 4. Heuristic: find contiguous code lines
    lines = response.split("\n")
    best_start = -1
    best_end = -1
    best_len = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(("def ", "import ", "from ", "class ")):
            start = i
            j = i + 1
            while j < len(lines):
                line = lines[j]
                if not line.strip():
                    # Allow blank lines within a block
                    j += 1
                    continue
                if line[0] in (" ", "\t"):
                    # Indented continuation
                    j += 1
                    continue
                stripped_j = line.lstrip()
                if stripped_j.startswith(("def ", "import ", "from ", "class ", "@")):
                    # New top-level definition, continue the block
                    j += 1
                    continue
                break
            # Trim trailing blank lines
            while j > start and not lines[j - 1].strip():
                j -= 1
            length = j - start
            if length > best_len:
                best_start = start
                best_end = j
                best_len = length
            i = j
        else:
            i += 1

    if best_len > 0:
        return "\n".join(lines[best_start:best_end]).strip()

    # 5. Fallback: entire response
    return response.strip()


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------


def execute_code_with_tests(
    code: str,
    test_list: list,
    test_setup_code: str = "",
    timeout: int = 10,
) -> dict:
    """
    Execute extracted code with test assertions in a subprocess.

    Returns dict with keys: passed (bool), error (str or None).
    """
    parts = []
    if test_setup_code and test_setup_code.strip():
        parts.append(test_setup_code.strip())
    parts.append(code)
    for test in test_list:
        parts.append(test)
    full_code = "\n".join(parts)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else None
        return {"passed": passed, "error": error}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP with LLaDA2.1 (KV-cache)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model Configuration --
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default="inclusionAI/LLaDA2.1-mini", help="HuggingFace model repo or local path")
    model_group.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype for model weights")
    model_group.add_argument("--device_map", type=str, default="auto", help="Transformers device_map value")
    model_group.add_argument("--cuda_visible_devices", "--cvd", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES before loading the model")
    model_group.add_argument("--mask_id", type=int, default=156895, help="Mask token id for diffusion")
    model_group.add_argument("--eos_id", type=int, default=156892, help="EOS token id for early stopping")

    # -- Generation Settings --
    gen_group = parser.add_argument_group("Generation Settings")
    gen_group.add_argument("--gen_length", type=int, default=512, help="Maximum generation length in tokens")
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

    # -- SSD-specific --
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

    # -- Evaluation Settings --
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--sample_n", type=int, default=10, help="Sample N test examples (0 or negative for full dataset)")
    eval_group.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    eval_group.add_argument("--no_chat", type=str2bool, default=False, help="Do not use tokenizer chat template")
    eval_group.add_argument("--verbose", type=str2bool, default=False, help="Print per-example generations and predictions")
    eval_group.add_argument("--prompt_length", type=int, default=4096, help="Maximum prompt length in tokens")
    eval_group.add_argument("--exec_timeout", type=int, default=10, help="Timeout in seconds for code execution")

    # -- Logging --
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_samples", type=str2bool, default=False, help="Save per-sample outputs and summary")
    log_group.add_argument("--output_path", type=str, default=None, help="Path to write results (json/jsonl)")
    log_group.add_argument("--summary_file", type=str, default=None, help="Append one JSONL record with config + final pass_at_1")
    log_group.add_argument("--config_str", type=str, default=None, help="Tag string appended to the summary record")

    # -- Other --
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    set_seed(args.seed)

    # -- Print configuration --
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
    print(f"  Few-shot: {args.num_fewshot}")
    print(f"  Exec timeout: {args.exec_timeout}s")
    print(f"  Seed: {args.seed}")

    # -- Load model --
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype_str=args.dtype,
        device_map=args.device_map,
    )

    # -- Load dataset --
    print("\nLoading MBPP dataset...")
    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")

    if args.sample_n is not None and args.sample_n > 0 and args.sample_n < len(dataset):
        rng = random.Random(123456)  # fixed for repeatability across runs
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        select = indices[: args.sample_n]
        select.sort()
        print(f"Selected indices (n={len(select)}): {select}")
        dataset = dataset.select(select)

    datasize = len(dataset)
    print("MBPP test size:", datasize)

    # -- Set up generate function and static kwargs --
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
    passed_so_far = 0
    total_nfe = 0
    pass_series = []

    eval_t0 = time.perf_counter()
    pbar = tqdm(range(datasize), desc="Evaluating")

    for idx in pbar:
        ex = dataset[idx]
        text = ex["text"]
        code_gold = ex["code"]
        test_list = ex["test_list"]
        test_setup_code = ex.get("test_setup_code", "")

        # Build prompt
        messages = build_chat_messages(text, test_list, num_fewshot=args.num_fewshot)

        if args.no_chat:
            # Flatten messages into a plain text prompt
            parts = [SYSTEM_PROMPT + "\n"]
            for msg in messages:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content']}")
            formatted = "\n\n".join(parts)
        else:
            # Prepend system message for chat template
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            try:
                formatted = tokenizer.apply_chat_template(
                    chat_messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                formatted = SYSTEM_PROMPT + "\n\n" + "\n\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in messages
                )

        input_ids = tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=args.prompt_length,
        )["input_ids"].to(model.device)

        generated_tokens, stats = gen_fn(
            model=model, input_ids=input_ids, **gen_fn_kwargs
        )
        total_nfe += stats["nfe"]

        out_text = tokenizer.decode(
            generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        extracted = extract_code(out_text)
        exec_result = execute_code_with_tests(
            extracted, test_list, test_setup_code=test_setup_code,
            timeout=args.exec_timeout,
        )
        passed = exec_result["passed"]
        if passed:
            passed_so_far += 1

        results.append(
            {
                "task_id": ex.get("task_id", idx),
                "text": text,
                "gold_code": code_gold,
                "model_output": out_text,
                "extracted_code": extracted,
                "passed": passed,
                "error": exec_result.get("error"),
            }
        )

        if args.verbose:
            print(f"\n[Example {idx}] task_id={ex.get('task_id', idx)}")
            print(f"Task: {text}")
            print(f"Model output:\n{out_text}")
            print(f"Extracted code:\n{extracted}")
            print(f"Passed: {passed}")
            if exec_result.get("error"):
                print(f"Error: {exec_result['error'][:500]}")

        processed = len(results)
        pass_rate = passed_so_far / float(processed)
        pass_series.append(pass_rate)
        pbar.set_postfix({"pass@1": f"{pass_rate:.4f}", "trend": ascii_sparkline(pass_series, max_len=80)})

    eval_seconds = time.perf_counter() - eval_t0

    cnt = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_at_1 = cnt / total if total > 0 else 0.0
    avg_nfe = total_nfe / total if total > 0 else 0.0
    print(f"pass@1: {cnt} / {total} = {pass_at_1:.4f}")
    print(f"Total NFE: {total_nfe}, Avg NFE: {avg_nfe:.1f}")

    results.append({"pass_at_1": pass_at_1, "total_nfe": total_nfe, "avg_nfe": avg_nfe})

    # -- Summary logging --
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
            "pass_at_1": pass_at_1,
            "passed": cnt,
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
            "num_fewshot": args.num_fewshot,
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
            f"pass_at_1={pass_at_1:.4f} ({cnt}/{total}) "
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
            f"num_fewshot={args.num_fewshot} "
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

    # -- Per-sample logging --
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

    if len(pass_series) > 1:
        print("pass@1 trend:")
        print(ascii_sparkline(pass_series, max_len=80))


if __name__ == "__main__":
    main()
