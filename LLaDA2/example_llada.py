"""
Minimal LLaDA2.1 inference example.

Usage:
  python example_llada.py
  python example_llada.py --model inclusionAI/LLaDA2.1-mini --prompt "Hello!"
  python example_llada.py --prompt_file path/to/prompt.txt --gen_length 1024
"""

import argparse
import sys
import time
from typing import Optional

import torch

from generate_utils import generate, generate_cached, generate_ssd_policy, load_model_and_tokenizer
from utils import set_seed, str2bool


DEFAULT_PROMPT = (
    "You are an expert Python programmer, and here is your task: Write a python function to remove "
    "first and last occurrence of a given character from the string. Your code should pass these tests:\n\n"
    'assert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\n'
    'assert remove_Occ("PHP","P") == "H"\n'
)


def build_text_input(tokenizer, prompt: str) -> str:
    """
    Prefer chat-template formatting if the tokenizer provides it;
    otherwise fall back to raw prompt.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def read_prompt_from_sources(prompt_arg, prompt_file, default_prompt):
    """Read prompt from CLI arg, file, stdin, or use default."""
    if prompt_arg is not None:
        return prompt_arg
    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data
    return default_prompt


@torch.no_grad()
def main(args):
    set_seed(args.seed)

    print(f"Loading model from: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype_str=args.dtype,
        device_map=args.device_map,
    )

    prompt = read_prompt_from_sources(args.prompt, args.prompt_file, DEFAULT_PROMPT)
    text = build_text_input(tokenizer, prompt)

    # Tokenize input
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    print(f"\n=== Prompt ===")
    print(prompt)
    print(f"\n=== Generating ({args.gen_length} tokens max) ===")

    # Generate
    t0 = time.perf_counter()
    stats = None

    # Common kwargs shared by all generate functions
    generate_fn_kwargs = dict(
        model=model,
        input_ids=input_ids,
        temperature=args.temperature,
        block_length=args.block_length,
        # steps=args.steps,
        gen_length=args.gen_length,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_early_stop=args.eos_early_stop,
        eos_id=args.eos_id,
        mask_id=args.mask_id,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        num_to_transfer=args.num_to_transfer,
        max_post_steps=args.max_post_steps,
        record_decoding_order=args.record_decoding_order,
        # minimal_topk=args.minimal_topk,
    )

    if args.generate_fn == "nocache":
        generated_tokens, stats = generate(**generate_fn_kwargs)
    elif args.generate_fn == "cached":
        generated_tokens, stats = generate_cached(**generate_fn_kwargs)
    elif args.generate_fn == "ssd_policy":
        generate_fn_kwargs.update(
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
        generated_tokens, stats = generate_ssd_policy(**generate_fn_kwargs)
    else:
        raise ValueError(f"Unknown generate function: {args.generate_fn}")

    t1 = time.perf_counter()

    # Decode output (generated_tokens includes prompt)
    prompt_len = input_ids.shape[1]
    output_text = tokenizer.decode(generated_tokens[0, prompt_len:], skip_special_tokens=True)
    num_generated = generated_tokens.shape[1] - prompt_len

    print(f"\n=== Output ===")
    print(output_text)
    print(f"\n=== Stats ===")
    print(f"Tokens generated: {num_generated}")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Speed: {num_generated / (t1 - t0):.1f} tok/s")
    print(f"NFE: {stats['nfe']}")

    if "decoding_order" in stats:
        print(f"Decoding steps: {len(stats['decoding_order'])}")

        gen_ids = generated_tokens[0].tolist()

        # Print position ordering per step
        for i, step_info in enumerate(stats["decoding_order"]):
            edits = step_info.get("edit", [])
            unmasks = step_info.get("unmask", [])
            if edits:
                edit_positions = [e[0] + 0.5 for e in edits]
                print(f"[order] step {i} <Edit>: {edit_positions}")
            if unmasks:
                print(f"[order] step {i} Unmask: {unmasks}")

        # Print decoded tokens per step
        for i, step_info in enumerate(stats["decoding_order"]):
            edits = step_info.get("edit", [])
            unmasks = step_info.get("unmask", [])
            if edits:
                items = []
                for abs_pos, old_tok, new_tok in edits:
                    old_piece = tokenizer.decode([old_tok], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
                    new_piece = tokenizer.decode([new_tok], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
                    items.append(f"{{{abs_pos}: {old_piece!r} -> {new_piece!r}}}")
                print(f"[decode] step {i} <Edit>: {', '.join(items)}")
            if unmasks:
                items = []
                for p in unmasks:
                    abs_pos = int(abs(p))
                    gen_pos = abs_pos - prompt_len
                    if 0 <= gen_pos < len(gen_ids):
                        tok_id = gen_ids[gen_pos]
                        piece = tokenizer.decode([tok_id], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
                        items.append(f"{{{p}: {piece!r}}}")
                if items:
                    print(f"[decode] step {i} Unmask: {', '.join(items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal LLaDA2.1 generation example")
    parser.add_argument("--model", type=str, default="inclusionAI/LLaDA2.1-mini", help="HuggingFace model repo or local path")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="User prompt text. If omitted, read from stdin/default.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a file containing the prompt.")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Torch dtype for model.")
    parser.add_argument("--device_map", type=str, default="auto", help="Transformers device_map value.")
    parser.add_argument("--gen_length", type=int, default=512, help="Number of tokens to generate.")
    parser.add_argument("--block_length", type=int, default=32, help="Block length used by model.generate.")
    # parser.add_argument("--steps", type=int, default=32, help="Refinement steps per block.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional nucleus sampling threshold.")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k sampling cutoff.")
    # parser.add_argument("--minimal_topk", type=int, default=1, help="Caps effective steps via gen_length // minimal_topk.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Acceptance threshold for generation.")
    parser.add_argument("--editing_threshold", type=float, default=0.5, help="Editing threshold for generation.")
    parser.add_argument("--max_post_steps", type=int, default=0, help="Post-mask global editing steps per block.")
    parser.add_argument("--eos_id", type=int, default=156892, help="EOS token id for early stopping.")
    parser.add_argument("--mask_id", type=int, default=156895, help="Mask token id used during iterative refinement.")
    parser.add_argument("--num_to_transfer", type=int, default=1, help="Minimum number of masked positions to resolve per iteration.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--eos_early_stop", type=str2bool, default=True, help="Enable/disable early stopping at EOS.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Generation function selection
    parser.add_argument("--generate_fn", type=str, choices=["nocache", "cached", "ssd_policy"], default="cached", help="Generation function to use.")
    parser.add_argument("--record_decoding_order", type=str2bool, default=False, help="Record and print per-step decoding order.")

    # SSD-specific arguments (used when --generate_fn=ssd_policy)
    parser.add_argument("--min_ssd_span_length", type=int, default=1, help="Minimum mask span length to trigger 2L verification.")
    parser.add_argument("--legacy_ssd_span_strategy", type=str2bool, default=False, help="If set, mask_span_length policy also checks high-confidence count before skipping verification.")
    parser.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0, help="Exponent applied to SSD acceptance ratios.")

    # SSD verification policy
    parser.add_argument("--do_verify_policy", type=str, default="mask_span_length",
                        choices=["mask_span_length", "score_threshold", "score_hysteresis"],
                        help="Policy for deciding whether to run the 2L verifier.")
    parser.add_argument("--do_verify_score_threshold", type=float, default=0.0, help="Threshold for score_threshold policy.")
    parser.add_argument("--hysteresis_threshold_on", type=float, default=0.0, help="Turn-on threshold for score_hysteresis policy.")
    parser.add_argument("--hysteresis_threshold_off", type=float, default=-1.0, help="Turn-off threshold for score_hysteresis policy.")
    parser.add_argument("--do_verify_score_type", type=str, default="difference_dynamic", choices=["difference_dynamic", "difference_static"],
                        help="Score function for score-based verify policies.")
    parser.add_argument("--score_penalty_coef", type=float, default=2.0, help="Penalty coefficient c in score computation.")
    parser.add_argument("--token_acceptance_estimator", type=str, default="hard_margin_threshold",
                        choices=["hard_margin_threshold", "soft_entropy_negexp", "soft_renyi_2_entropy"],
                        help="Estimator for per-token acceptance probability.")
    parser.add_argument("--ssd_confidence_margin_threshold", type=float, default=0.05, help="Margin threshold for hard_margin_threshold estimator.")
    parser.add_argument("--ssd_entropy_temperature", type=float, default=1.0, help="Temperature for soft_entropy_negexp estimator.")

    args = parser.parse_args()
    main(args)
