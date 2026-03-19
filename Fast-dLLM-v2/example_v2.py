"""
Minimal Fast-dLLM v2 inference example.

Usage:
  python v2/example_v2.py
  python v2/example_v2.py --model Efficient-Large-Model/Fast_dLLM_v2_7B --prompt "Hello!"
  python v2/example_v2.py --prompt "Hello!" --no_chat true
"""

import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_utils import generate, generate_ssd
from generate_policy_utils import generate_ssd_policy
from eval_gsm8k_fast_v2 import set_seed, str2bool


def build_text_input(tokenizer, prompt: str, use_chat_template: bool = True) -> str:
    """
    Prefer chat-template formatting if the tokenizer provides it;
    otherwise fall back to raw prompt.
    """
    if not use_chat_template:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


@torch.no_grad()
def main(args):
    set_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {args.model} (device_map={device})")
    if args.use_local_modeling:
        from modeling_fast import Fast_dLLM_QwenForCausalLM
        model_cls = Fast_dLLM_QwenForCausalLM
        print("  Using local modeling file: modeling_fast.py")
    else:
        model_cls = AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    prompt = args.prompt
    if prompt is None:
        # prompt = (
        #     "Question: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk and take care of their "
        #     "business. How many hours a week does he spend taking care of dogs?\n"
        #     "Please reason step by step, and put your final answer within \\boxed{}.\n"
        # )
        prompt = (
            "You are an expert Python programmer, and here is your task: Write a python function to remove "
            "first and last occurrence of a given character from the string. Your code should pass these tests:\n\n"
            'assert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\n'
            'assert remove_Occ("PHP","P") == "H"\n[BEGIN]\n'
        )

    text = build_text_input(tokenizer, prompt, use_chat_template=not args.no_chat)
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build generation kwargs from args
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        use_block_cache=args.use_block_cache,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        return_decoding_order=args.return_decoding_order,
        use_attention_mask=args.use_attention_mask,
    )
    if args.generate_fn == "ssd":
        gen_kwargs.update(
            use_ssd_cache=args.use_ssd_cache,
            ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
            min_ssd_span_length=args.min_ssd_span_length,
            allow_resample=args.allow_resample,
            cache_ver=args.cache_ver,
            draft_ver=args.draft_ver,
        )
    elif args.generate_fn == "ssd_policy":
        gen_kwargs.update(
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

    if args.generate_fn == "ssd":
        gen_fn = generate_ssd
    elif args.generate_fn == "ssd_policy":
        gen_fn = generate_ssd_policy
    else:
        gen_fn = generate
    generated = gen_fn(model, inputs["input_ids"], tokenizer=tokenizer, **gen_kwargs)

    stats = None
    if isinstance(generated, tuple):
        generated, stats = generated

    seq = generated.sequences if hasattr(generated, "sequences") else generated
    prompt_len = inputs["input_ids"].shape[1]
    out_ids = seq[0][prompt_len:]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Output ===")
    print(out_text)
    if stats is not None:
        print("\n=== Decoding stats ===")
        print(f"total_forward_steps: {stats.get('total_forward_steps')}")
        decoding_order = stats.get("decoding_order") or []
        print(f"decoding_order_steps: {len(decoding_order)}")
        for i, step in enumerate(decoding_order[:]):
            print(f"step {i}: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Fast-dLLM v2 generation example")
    parser.add_argument("--model", type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B", help="HF model name or local path")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--no_chat", type=str2bool, default=False,
                        help="If true, skip chat template and use raw prompt text (lm-eval style).")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--generate_fn", type=str, default="fast", choices=["fast", "ssd", "ssd_policy"],
                        help='Which generator to use: "fast" (block diffusion), "ssd" (self-speculative decoding), or "ssd_policy" (SSD with configurable verify policy).')
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--use_block_cache", type=str2bool, default=False)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_ssd_cache", type=str2bool, default=False, help="(SSD only) Enable intra-block SSD cache patching.")
    parser.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0, help="(SSD only) Tempering factor for q/p ratios.")
    parser.add_argument("--min_ssd_span_length", type=int, default=1, help="(SSD only) Min contiguous span length for SSD verify.")
    parser.add_argument("--allow_resample", type=str2bool, default=True, help="(SSD only) Residual-resample at first rejected token.")
    parser.add_argument("--return_decoding_order", type=str2bool, default=False, help="Record per-step decoding order.")
    parser.add_argument("--cache_ver", type=str2bool, default=False, help="Enable cache verification.")
    parser.add_argument("--draft_ver", type=str2bool, default=False, help="Enable draft verification.")
    parser.add_argument("--do_verify_policy", type=str, default="mask_span_length",
                        choices=["mask_span_length", "score_threshold", "score_hysteresis", "contextual_bandit_ucb"],
                        help="(SSD policy only) Policy for deciding whether to run verification.")
    parser.add_argument("--do_verify_score_threshold", type=float, default=0.0,
                        help="(SSD policy only) Threshold for score_threshold policy.")
    parser.add_argument("--hysteresis_threshold_on", type=float, default=0.0,
                        help="(SSD policy only) Turn-on threshold for score_hysteresis.")
    parser.add_argument("--hysteresis_threshold_off", type=float, default=-1.0,
                        help="(SSD policy only) Turn-off threshold for score_hysteresis.")
    parser.add_argument("--token_acceptance_estimator", type=str, default="hard_margin_threshold",
                        choices=["hard_margin_threshold", "soft_entropy_negexp", "soft_renyi_2_entropy"],
                        help="(SSD policy only) Estimator used by score-based policies.")
    parser.add_argument("--ssd_confidence_margin_threshold", type=float, default=0.05,
                        help="(SSD policy only) Margin threshold for hard_margin_threshold estimator.")
    parser.add_argument("--ssd_entropy_temperature", type=float, default=1.0,
                        help="(SSD policy only) Temperature for soft_entropy_negexp estimator.")
    parser.add_argument("--ucb_beta", type=float, default=1.0,
                        help="(SSD policy only) Exploration coefficient for contextual_bandit_ucb.")
    parser.add_argument("--ucb_span_length_bins", type=int, default=2,
                        help="(SSD policy only) Number of span-length bins for UCB context.")
    parser.add_argument("--ucb_block_progress_bins", type=int, default=2,
                        help="(SSD policy only) Number of block-progress bins for UCB context.")
    parser.add_argument("--ucb_entropy_bins", type=int, default=2,
                        help="(SSD policy only) Number of entropy bins for UCB context.")
    parser.add_argument("--ucb_entropy_source", type=str, default="span", choices=["span", "masked"],
                        help="(SSD policy only) Entropy source for UCB context features.")
    parser.add_argument("--do_verify_score_type", type=str, default="difference_dynamic",
                        choices=["difference_dynamic", "difference_static"],
                        help="(SSD policy only) Score function for score-based policies.")
    parser.add_argument("--score_penalty_coef", type=float, default=2.0,
                        help="(SSD policy only) Penalty coefficient used by score function.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help='Device mapping, e.g. "cuda:0" or "cpu". Default: auto-detect.')
    parser.add_argument("--use_local_modeling", type=str2bool, default=True,
                        help="Load model class from local v2/modeling_fast.py instead of HF remote code.")
    parser.add_argument("--use_attention_mask", type=str2bool, default=True, help="Use attention mask.")
    args = parser.parse_args()

    main(args)
