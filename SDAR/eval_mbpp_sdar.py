#!/usr/bin/env python3
"""
Evaluate an SDAR (block diffusion) model on MBPP.

Uses `block_diffusion_generate()` from `generate.py` / `generate_ssd.py` /
`generate_ssd_policy.py` for inference, with subprocess-based test execution.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
# Code extraction
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
                    j += 1
                    continue
                if line[0] in (" ", "\t"):
                    j += 1
                    continue
                stripped_j = line.lstrip()
                if stripped_j.startswith(("def ", "import ", "from ", "class ", "@")):
                    j += 1
                    continue
                break
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


def _normalize_stop_ids(tokenizer, stopping_criteria_idx):
    if stopping_criteria_idx is None:
        return None
    if isinstance(stopping_criteria_idx, int):
        return [stopping_criteria_idx]
    return list(stopping_criteria_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP with an SDAR block-diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dir", default="JetLM/SDAR-8B-Chat", type=str, help="Path to pretrained model directory")
    model_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights",
    )
    model_group.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for HF loading (e.g. "auto").',
    )
    model_group.add_argument(
        "--cuda-visible-devices",
        "--cvd",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES before loading the model",
    )
    model_group.add_argument("--mask_id", type=int, default=None, help="Mask token id for diffusion")

    # Generation settings (SDAR)
    gen_group = parser.add_argument_group("Generation Settings (SDAR)")
    gen_group.add_argument("--prompt_length", type=int, default=4096, help="Maximum prompt length in tokens")
    gen_group.add_argument("--gen_length", type=int, default=512, help="Maximum generation length in tokens")
    gen_group.add_argument("--block_length", type=int, default=4, help="Block length for diffusion decoding")
    gen_group.add_argument("--denoising_steps", type=int, default=4, help="Number of denoising steps per block")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (>0)")
    gen_group.add_argument("--top_k", type=int, default=0, help="Top-K sampling (0 disables)")
    gen_group.add_argument("--top_p", type=float, default=1.0, help="Top-P sampling probability threshold")
    gen_group.add_argument(
        "--remasking_strategy",
        type=str,
        default="low_confidence_static",
        choices=["low_confidence_dynamic", "low_confidence_static", "sequential", "entropy_bounded"],
        help="Strategy for remasking tokens",
    )
    gen_group.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for low-confidence dynamic remasking",
    )
    gen_group.add_argument(
        "--eb_threshold",
        type=float,
        default=0.35,
        help="Entropy threshold for entropy bounded sampling",
    )
    gen_group.add_argument(
        "--stopping_criteria_idx",
        type=int,
        nargs="+",
        default=None,
        help="List of token IDs that stop generation (e.g. eos_token_id)",
    )
    gen_group.add_argument("--generate_fn", type=str, default="bd3", choices=["bd3", "ssd", "ssd_policy", "ssd2"], help="Function to generate text")
    gen_group.add_argument("--cache_ver", type=str2bool, default=False, help="Use cache verification")
    gen_group.add_argument("--draft_ver", type=str2bool, default=False, help="Use draft verification")
    gen_group.add_argument("--min_ssd_span_length", type=int, default=1, help="Minimum length of span to invoke SSD")
    gen_group.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0, help="Tempering factor for SSD ratio")
    gen_group.add_argument("--always_check_high_confidence", type=str2bool, default=False, help="If set, always check high confidence tokens for remaining masked positions")

    # SSD Policy & Parameters (SSD/SDAR)
    ssd_group = parser.add_argument_group("SSD Policy Settings")
    ssd_group.add_argument("--do_verify_policy", type=str, default="mask_span_length",
                           choices=["mask_span_length", "score_threshold", "score_hysteresis", "contextual_bandit_ucb"],
                           help="Policy for verifying whether to invoke SSD (Self-Speculative Decoding)")
    ssd_group.add_argument("--do_verify_score_threshold", type=float, default=0.0,
                           help="Threshold for score_threshold policy: do_verify_score < do_verify_score_threshold")
    ssd_group.add_argument("--hysteresis_threshold_on", type=float, default=0.0,
                           help="Hysteresis turn-on threshold: if s > th_on then do verify")
    ssd_group.add_argument("--hysteresis_threshold_off", type=float, default=-1.0,
                           help="Hysteresis turn-off threshold: if s < th_off then stop verify")
    ssd_group.add_argument("--do_verify_score_type", type=str, default="difference_dynamic",
                           choices=["difference_dynamic", "difference_static"],
                           help="Scoring function used by score-based verify policies")
    ssd_group.add_argument("--score_penalty_coef", type=float, default=2.0,
                           help="Penalty coefficient c used by do_verify_score_type")
    ssd_group.add_argument("--ucb_beta", type=float, default=1.0,
                           help="Exploration coefficient beta for contextual_bandit_ucb policy")
    ssd_group.add_argument("--ucb_span_length_bins", type=int, default=2,
                           help="Number of span-length bins in contextual_bandit_ucb policy")
    ssd_group.add_argument("--ucb_block_progress_bins", type=int, default=2,
                           help="Number of block-progress bins in contextual_bandit_ucb policy")
    ssd_group.add_argument("--ucb_entropy_bins", type=int, default=2,
                           help="Number of entropy bins in contextual_bandit_ucb policy")
    ssd_group.add_argument("--ucb_entropy_source", type=str, default="span",
                           choices=["span", "masked"],
                           help="Entropy source for contextual_bandit_ucb policy: first span or all masked tokens")
    ssd_group.add_argument("--token_acceptance_estimator", type=str, default="hard_margin_threshold",
                           choices=[
                               "hard_margin_threshold",
                               "hard_entropy_threshold",
                               "soft_confidence_power",
                               "soft_clipped_linear_margin",
                               "soft_entropy_negexp",
                               "soft_renyi_2_entropy",
                           ],
                           help="Estimator for per-token acceptance probability alpha_i used by SSD expected-acceptance estimation")
    ssd_group.add_argument("--ssd_confidence_margin_threshold", type=float, default=0.05,
                           help="Threshold for hard_margin_threshold estimator: alpha_i = 1[m_i > threshold]")
    ssd_group.add_argument("--ssd_confidence_power", type=float, default=0.8,
                           help="Power for soft_confidence_power estimator: alpha_i = confidence_i ** power")
    ssd_group.add_argument("--ssd_entropy_threshold", type=float, default=0.1,
                           help="Threshold for hard_entropy_threshold estimator: alpha_i = 1[H_i < threshold], H_i is normalized entropy")
    ssd_group.add_argument("--ssd_confidence_margin_coef", type=float, default=1.0,
                           help="Coefficient for soft_clipped_linear_margin estimator: alpha_i = clip(coef * margin_i, 0, 1)")
    ssd_group.add_argument("--ssd_entropy_temperature", type=float, default=1.0,
                           help="Temperature for soft_entropy_negexp estimator: alpha_i = exp(-temp * H_i)")
    ssd_group.add_argument("--legacy_ssd_span_strategy", type=str2bool, default=False, help="If set, use legacy SSD span strategy")
    ssd_group.add_argument("--allow_resample", type=str2bool, default=True, help="If set, allow resampling of rejected tokens")

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--limit", type=float, default=10, help="Limit number of examples (>=1: count, <1: ratio of dataset, None/0: full)")
    eval_group.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    eval_group.add_argument("--no_chat", type=str2bool, default=False, help="Do not use tokenizer chat template")
    eval_group.add_argument("--verbose", type=str2bool, default=False, help="Print per-example generations and predictions")
    eval_group.add_argument("--exec_timeout", type=int, default=10, help="Timeout in seconds for code execution")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_samples", type=str2bool, default=False, help="Save per-sample outputs and summary")
    log_group.add_argument("--output_path", type=str, default=None, help="Path to write results (json/jsonl)")
    log_group.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="If set, append one JSONL record with config + final pass@1 to this file",
    )
    log_group.add_argument("--config_str", type=str, default=None, help="If set, append this string to the summary file")

    # Other settings
    other_group = parser.add_argument_group("Other Settings")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    # Keep this flag in argparse for compatibility, but force-disable it.
    args.legacy_ssd_span_strategy = False

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.gen_length <= 0:
        parser.error("--gen_length must be positive")
    if args.block_length <= 0:
        parser.error("--block_length must be positive")
    if args.denoising_steps < 0:
        parser.error("--denoising_steps must be non-negative")
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error("--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic")
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error("--eb_threshold is required when --remasking_strategy=entropy_bounded")

    set_seed(args.seed)

    print("\nConfiguration:")
    print(f"  Model: {args.model_dir}")
    print(f"  Generate function: {args.generate_fn}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Device map: {args.device_map}")
    print(f"  Mask id: {args.mask_id}")
    print(f"  Prompt length: {args.prompt_length}")
    print(f"  Gen length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Denoising steps: {args.denoising_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Remasking: {args.remasking_strategy}")
    print(f"  Few-shot: {args.num_fewshot}")
    print(f"  No chat: {args.no_chat}")
    print(f"  Exec timeout: {args.exec_timeout}s")
    print(f"  Seed: {args.seed}")

    if args.generate_fn == "bd3":
        from generate import block_diffusion_generate
        generate_fn = block_diffusion_generate
        generate_fn_kwargs = {}
    elif args.generate_fn == "ssd":
        from generate_ssd import block_diffusion_generate
        generate_fn = block_diffusion_generate
        generate_fn_kwargs = {
            'cache_ver': args.cache_ver,
            'draft_ver': args.draft_ver,
            'min_ssd_span_length': args.min_ssd_span_length,
            'ssd_ratio_tempering_factor': args.ssd_ratio_tempering_factor,
            'always_check_high_confidence': args.always_check_high_confidence,
            'legacy_ssd_span_strategy': args.legacy_ssd_span_strategy,
            'allow_resample': args.allow_resample,
        }
    elif args.generate_fn == "ssd_policy":
        from generate_ssd_policy import block_diffusion_generate
        generate_fn = block_diffusion_generate
        generate_fn_kwargs = {
            'cache_ver': args.cache_ver,
            'draft_ver': args.draft_ver,
            'min_ssd_span_length': args.min_ssd_span_length,
            'ssd_ratio_tempering_factor': args.ssd_ratio_tempering_factor,
            'always_check_high_confidence': args.always_check_high_confidence,
            'legacy_ssd_span_strategy': args.legacy_ssd_span_strategy,
            'allow_resample': args.allow_resample,
            'do_verify_policy': args.do_verify_policy,
            'do_verify_score_threshold': args.do_verify_score_threshold,
            'hysteresis_threshold_on': args.hysteresis_threshold_on,
            'hysteresis_threshold_off': args.hysteresis_threshold_off,
            'token_acceptance_estimator': args.token_acceptance_estimator,
            'ssd_confidence_margin_threshold': args.ssd_confidence_margin_threshold,
            'ssd_confidence_power': args.ssd_confidence_power,
            'ssd_entropy_threshold': args.ssd_entropy_threshold,
            'ssd_confidence_margin_coef': args.ssd_confidence_margin_coef,
            'ssd_entropy_temperature': args.ssd_entropy_temperature,
            'ucb_beta': args.ucb_beta,
            'ucb_span_length_bins': args.ucb_span_length_bins,
            'ucb_block_progress_bins': args.ucb_block_progress_bins,
            'ucb_entropy_bins': args.ucb_entropy_bins,
            'ucb_entropy_source': args.ucb_entropy_source,
            'do_verify_score_type': args.do_verify_score_type,
            'score_penalty_coef': args.score_penalty_coef,
        }
    else:
        raise ValueError(f"Invalid generate function: {args.generate_fn}")

    print("\nLoading model and tokenizer...")
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if args.mask_id is None:
        if tokenizer.mask_token is None:
            parser.error("--mask_id is required because tokenizer.mask_token is None")
        args.mask_id = tokenizer(tokenizer.mask_token)["input_ids"][0]

    if args.stopping_criteria_idx is None:
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
            args.stopping_criteria_idx = gen_cfg.eos_token_id
        except Exception:
            args.stopping_criteria_idx = None
    args.stopping_criteria_idx = _normalize_stop_ids(tokenizer, args.stopping_criteria_idx)

    print("\nLoading MBPP dataset...")
    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")

    if args.limit is not None and args.limit > 0:
        if args.limit < 1:
            sample_n = max(1, int(len(dataset) * args.limit))
        else:
            sample_n = int(args.limit)
        if sample_n < len(dataset):
            rng = random.Random(args.seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            select = indices[:sample_n]
            select.sort()
            print(f"Selected indices (n={len(select)}): {select}")
            dataset = dataset.select(select)

    datasize = len(dataset)
    print("MBPP test size:", datasize)

    results = []
    passed_so_far = 0
    pass_series = []

    mask_token_str = tokenizer.mask_token or "<|MASK|>"
    stop_ids = args.stopping_criteria_idx

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
            user_content = build_user_message(text, test_list)
            formatted = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_content}\n<|assistant|>\n"
        else:
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            try:
                formatted = tokenizer.apply_chat_template(
                    chat_messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                formatted = SYSTEM_PROMPT + "\n\n" + "\n\n".join(
                    f"{m['role'].capitalize()}: {m['content']}" for m in messages
                )

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

        # Reseed per example for reproducibility
        set_seed(args.seed + idx * 100000)

        out_ids = generate_fn(
            model,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            remasking_strategy=args.remasking_strategy,
            confidence_threshold=args.confidence_threshold,
            eb_threshold=args.eb_threshold,
            stopping_criteria_idx=stop_ids,
            **generate_fn_kwargs,
        )

        gen_ids = out_ids[:, prompt_len : prompt_len + args.gen_length]
        out_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        out_text = out_text.replace(mask_token_str, "").replace("<|MASK|>", "")

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
    print(f"pass@1: {cnt} / {total} = {pass_at_1:.4f}")

    results.append({"pass_at_1": pass_at_1})

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
            "eval_seconds": eval_seconds,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "temperature": args.temperature,
            "remasking_strategy": args.remasking_strategy,
            "confidence_threshold": args.confidence_threshold,
            "generate_fn": args.generate_fn,
            "cache_ver": args.cache_ver,
            "draft_ver": args.draft_ver,
            "min_ssd_span_length": args.min_ssd_span_length,
            "ssd_ratio_tempering_factor": args.ssd_ratio_tempering_factor,
            "always_check_high_confidence": args.always_check_high_confidence,
            "model_dir": args.model_dir,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "seed": args.seed,
            "argv": list(sys.argv),
        }
        if args.generate_fn == "ssd_policy":
            summary_rec.update(
                {
                    "legacy_ssd_span_strategy": args.legacy_ssd_span_strategy,
                    "allow_resample": args.allow_resample,
                    "do_verify_policy": args.do_verify_policy,
                    "do_verify_score_threshold": args.do_verify_score_threshold,
                    "hysteresis_threshold_on": args.hysteresis_threshold_on,
                    "hysteresis_threshold_off": args.hysteresis_threshold_off,
                    "token_acceptance_estimator": args.token_acceptance_estimator,
                    "ssd_confidence_margin_threshold": args.ssd_confidence_margin_threshold,
                    "ssd_confidence_power": args.ssd_confidence_power,
                    "ssd_entropy_threshold": args.ssd_entropy_threshold,
                    "ssd_confidence_margin_coef": args.ssd_confidence_margin_coef,
                    "ssd_entropy_temperature": args.ssd_entropy_temperature,
                    "ucb_beta": args.ucb_beta,
                    "ucb_span_length_bins": args.ucb_span_length_bins,
                    "ucb_block_progress_bins": args.ucb_block_progress_bins,
                    "ucb_entropy_bins": args.ucb_entropy_bins,
                    "ucb_entropy_source": args.ucb_entropy_source,
                    "do_verify_score_type": args.do_verify_score_type,
                    "score_penalty_coef": args.score_penalty_coef,
                }
            )
        with open(summary_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

        txt_line = (
            f"config={summary_rec['config']} "
            f"pass_at_1={pass_at_1:.4f} ({cnt}/{total}) "
            f"eval_seconds={eval_seconds:.3f} "
            f"gen_length={args.gen_length} "
            f"block_length={args.block_length} "
            f"denoising_steps={args.denoising_steps} "
            f"temperature={args.temperature} "
            f"remasking_strategy={args.remasking_strategy} "
            f"confidence_threshold={args.confidence_threshold} "
            f"generate_fn={args.generate_fn} "
            f"cache_ver={int(bool(args.cache_ver))} "
            f"draft_ver={int(bool(args.draft_ver))} "
            f"min_ssd_span_length={args.min_ssd_span_length} "
            f"ssd_ratio_tempering_factor={args.ssd_ratio_tempering_factor} "
            f"always_check_high_confidence={int(bool(args.always_check_high_confidence))} "
            f"model_dir={args.model_dir} "
            f"limit={args.limit} "
            f"num_fewshot={args.num_fewshot} "
            f"seed={args.seed}"
        )
        if args.generate_fn == "ssd_policy":
            txt_line += (
                f" legacy_ssd_span_strategy={int(bool(args.legacy_ssd_span_strategy))}"
                f" allow_resample={int(bool(args.allow_resample))}"
                f" do_verify_policy={args.do_verify_policy}"
                f" do_verify_score_threshold={args.do_verify_score_threshold}"
                f" hysteresis_threshold_on={args.hysteresis_threshold_on}"
                f" hysteresis_threshold_off={args.hysteresis_threshold_off}"
                f" token_acceptance_estimator={args.token_acceptance_estimator}"
                f" ssd_confidence_margin_threshold={args.ssd_confidence_margin_threshold}"
                f" ssd_confidence_power={args.ssd_confidence_power}"
                f" ssd_entropy_threshold={args.ssd_entropy_threshold}"
                f" ssd_confidence_margin_coef={args.ssd_confidence_margin_coef}"
                f" ssd_entropy_temperature={args.ssd_entropy_temperature}"
                f" ucb_beta={args.ucb_beta}"
                f" ucb_span_length_bins={args.ucb_span_length_bins}"
                f" ucb_block_progress_bins={args.ucb_block_progress_bins}"
                f" ucb_entropy_bins={args.ucb_entropy_bins}"
                f" ucb_entropy_source={args.ucb_entropy_source}"
                f" do_verify_score_type={args.do_verify_score_type}"
                f" score_penalty_coef={args.score_penalty_coef}"
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
