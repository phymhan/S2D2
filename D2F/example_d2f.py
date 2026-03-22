import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
from typing import Dict, Set

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig

# Local model cache wrappers
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig
from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    :param v:
    :return: bool(v)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_full_block_attention_mask(prompt_length: int, max_length: int, block_size: int, device=None, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    # Prompt attends to itself
    attention_mask[:, :, :prompt_length, :prompt_length] = 0

    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        # Current block can see the prompt
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        # Current block can see all previous regular blocks
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        # Current block can see itself
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask


def extract_attention_mask(full_mask: torch.Tensor, start_pos: int, input_length: int, cache_length: int) -> torch.Tensor:
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, device=full_mask.device, dtype=full_mask.dtype)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted_mask


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0, top_p: float = None, top_k: int = None):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    return initial_confidence, x0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Simple D2F block-based generation example")
    parser.add_argument(
        '--model',
        type=str,
        default='llada',
        choices=['llada', 'dream'],
        help='Which base model family to use (loads default base+LoRA paths)',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--block_add_threshold', type=float, default=0.5)
    parser.add_argument('--decoded_token_threshold', type=float, default=0.8)
    parser.add_argument('--skip_threshold', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--post_rope_cache', type=str2bool, default=False,
                        help='Enable post-RoPE KV caching if supported by the model config')
    parser.add_argument('--truncate_at_eos', type=str2bool, default=True,
                        help='If true, truncate generated tokens at first EOS')
    parser.add_argument('--prompt', type=str, default='<dog>',
                        help='Prompt text or one of <dog>, <lily>, <quantum>, <count>, <code>')
    parser.add_argument('--no_chat', type=str2bool, default=False,
                        help='If true, do not apply chat template')
    parser.add_argument('--forward_stats', type=str2bool, default=False,
                        help='Collect and print per-block forward counts and averages (single-sample only)')
    parser.add_argument('--generate_fn', type=str, default='d2f', choices=['d2f', 'ssd', 'ssd_old'], help='Generation function to use')
    # SSD / self-speculative decoding options (LLaDA supports this; Dream support is experimental)
    parser.add_argument('--target_block_size', type=int, default=1)
    parser.add_argument('--ssd_partial_ar_span', type=int, default=0)
    parser.add_argument('--min_tokens_per_step', type=int, default=1)
    parser.add_argument('--ssd_threshold_draft_confidence', type=str2bool, default=False)
    parser.add_argument('--draft_confidence_threshold', type=float, default=None)
    parser.add_argument('--verify_confidence_threshold', type=float, default=None)
    parser.add_argument('--min_ssd_span_length', type=int, default=1)
    parser.add_argument('--cache_ver', type=str2bool, default=False)
    parser.add_argument('--draft_ver', type=str2bool, default=False)
    parser.add_argument('--allow_resample', type=str2bool, default=True)
    parser.add_argument('--ssd_ratio_tempering_factor', type=float, default=1.0)
    # Debug mode
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        target_dtype = torch.bfloat16
    elif args.dtype == 'float16':
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    # Load model + LoRA
    if args.model == "llada":
        pretrained_path = "GSAI-ML/LLaDA-8B-Instruct"
        lora_path = "SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora"
        config = LLaDAConfig.from_pretrained(pretrained_path)
        try:
            # Overwrite config option before model init so blocks see it
            setattr(config, 'post_rope_cache', bool(args.post_rope_cache))
        except Exception:
            pass
        model = LLaDAModelLM.from_pretrained(
            pretrained_path, config=config, torch_dtype=target_dtype
        ).eval()
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.to(device)
        mask_token_id = 126336
    else:
        # Defaults from `eval_dream.sh`
        pretrained_path = "Dream-org/Dream-v0-Base-7B"
        lora_path = "SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"
        config = DreamConfig.from_pretrained(pretrained_path)
        model = DreamModel.from_pretrained(
            pretrained_path, config=config, torch_dtype=target_dtype, trust_remote_code=True
        ).eval()
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.to(device)
        mask_token_id = int(getattr(config, "mask_token_id", 151666))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.truncate_at_eos:
        if args.model == "dream":
            # Dream/Qwen-style chat uses <|im_end|> as an end-of-turn marker.
            # Also keep eos (<|endoftext|>) as a fallback stop id.
            gen_cfg = GenerationConfig.from_pretrained(pretrained_path)
            stop_ids = []
            if gen_cfg.eos_token_id is not None:
                stop_ids = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
            try:
                im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
                if im_start_id is not None and im_start_id >= 0:
                    stop_ids.append(int(im_start_id))
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is not None and im_end_id >= 0:
                    stop_ids.append(int(im_end_id))
            except Exception:
                pass
            # de-dup, preserve order
            seen = set()
            eos_token_id = [x for x in stop_ids if not (x in seen or seen.add(x))]
        else:
            eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = None

    # Build prompt
    if args.prompt in ("<dog>", "<lily>", "<quantum>", "<count>", "<code>"):
        if args.prompt in ("<dog>", "<lily>"):
            dataset = load_dataset('gsm8k', 'main')
            fewshot = ''
            for i in range(5):
                fewshot += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
            if args.prompt == "<dog>":
                tail = "Question: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
            else:
                tail = "Question: Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
            # prompt = fewshot + tail
            prompt = tail
        elif args.prompt == "<quantum>":
            prompt = "Explain in detail what is quantum computing and how it works, what is the advantage of quantum computing over classical computing? Please reason step by step."
        elif args.prompt == "<code>":
            prompt = "Write PyTorch code to implement a transformer block with self-attention. Please reason step by step."
        else:
            prompt = "Please count down from 1000 to 1. Do not omit any numbers, use space to separate each number."
    else:
        prompt = args.prompt
    
    if args.debug:
        dataset = load_dataset('gsm8k', 'main')
        fewshot = ''
        for i in range(1):
            fewshot += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
        prompt = fewshot + 'Question: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?\nFirst, answer the question. Then Explain in detail what is quantum computing and how it works, what is the advantage of quantum computing over classical computing? Please reason step by step.'

    # Dream evaluator prepends BOS token to the prompt string.
    # if args.model == "dream" and getattr(tokenizer, "bos_token", None):
    #     if not prompt.startswith(tokenizer.bos_token):
    #         prompt = tokenizer.bos_token + prompt

    if args.no_chat:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        chat_history = [{"role": "user", "content": prompt}]
        messages = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        if args.model == "dream":
            # For Qwen-style chat templates, the template already includes special tokens.
            inputs = tokenizer(messages, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        else:
            inputs = tokenizer(messages, return_tensors="pt").input_ids.to(device)

    # Truncate left if needed
    if inputs.shape[1] > args.max_length - args.max_new_tokens:
        inputs = inputs[:, -(args.max_length - args.max_new_tokens):]

    if args.batch_size > 1:
        inputs = inputs.repeat(args.batch_size, 1)

    start_time = time.time()

    forward_stats = None
    # SDAR-style dispatch: select a function and pass extra kwargs.
    if args.generate_fn == "ssd":
        from generation_utils import generate_block_speculative
        generate_fn = generate_block_speculative
        generate_fn_kwargs = {
            "target_block_size": args.target_block_size,
            "ssd_partial_ar_span": args.ssd_partial_ar_span,
            "min_tokens_per_step": args.min_tokens_per_step,
            "ssd_threshold_draft_confidence": bool(args.ssd_threshold_draft_confidence),
            "min_ssd_span_length": args.min_ssd_span_length,
            "cache_ver": bool(args.cache_ver),
            "draft_ver": bool(args.draft_ver),
            "allow_resample": bool(args.allow_resample),
            "ssd_ratio_tempering_factor": args.ssd_ratio_tempering_factor,
            "draft_confidence_threshold": args.draft_confidence_threshold,
            "verify_confidence_threshold": args.verify_confidence_threshold,
        }
    elif args.generate_fn == "ssd_old":
        from generation_utils_old import generate_block_speculative
        generate_fn = generate_block_speculative
        generate_fn_kwargs = {
            "target_block_size": args.target_block_size,
            "ssd_partial_ar_span": args.ssd_partial_ar_span,
            "ssd_min_tokens_per_step": args.min_tokens_per_step,
            "ssd_threshold_draft_confidence": bool(args.ssd_threshold_draft_confidence),
            "min_ssd_span_length": args.min_ssd_span_length,
            "cache_ver": bool(args.cache_ver),
            "draft_ver": bool(args.draft_ver),
            "allow_resample": bool(args.allow_resample),
            "ssd_ratio_tempering_factor": args.ssd_ratio_tempering_factor,
            "draft_confidence_threshold": args.draft_confidence_threshold,
            "verify_confidence_threshold": args.verify_confidence_threshold,
        }
    else:
        from generation_utils import generate_block_single
        generate_fn = generate_block_single
        generate_fn_kwargs = {}

    gen_ids = generate_fn(
        model,
        inputs,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        mask_token_id=mask_token_id,
        model_type=args.model,
        eos_token_id=eos_token_id,
        device=device,
        dtype=target_dtype if target_dtype is not None else torch.bfloat16,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        block_add_threshold=args.block_add_threshold,
        decoded_token_threshold=args.decoded_token_threshold,
        skip_threshold=args.skip_threshold,
        margin_confidence=False,
        neg_entropy=False,
        return_forward_stats=args.forward_stats,
        **generate_fn_kwargs,
    )
    if args.forward_stats:
        gen_ids, forward_stats = gen_ids

    elapsed = time.time() - start_time

    if isinstance(gen_ids, torch.Tensor) and gen_ids.dim() == 2:
        texts = [tokenizer.decode(gen_ids[i], skip_special_tokens=False) for i in range(gen_ids.size(0))]
        for i, t in enumerate(texts):
            print(f"[sample {i}] {t}")
        total_tokens = sum(int(gen_ids[i].numel()) for i in range(gen_ids.size(0)))
        print(f"\n[info] Generated {total_tokens} tokens across {gen_ids.size(0)} samples in {elapsed:.2f}s ({total_tokens/max(elapsed,1e-6):.2f} tok/s)")
    else:
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        print(text)
        tok_count = len(gen_ids) if not isinstance(gen_ids, torch.Tensor) else int(gen_ids.numel())
        print(f"\n[info] Generated {tok_count} tokens in {elapsed:.2f}s ({tok_count/max(elapsed,1e-6):.2f} tok/s)")
        if forward_stats is not None:
            per_block = forward_stats.get('per_block_forwards', {})
            avg_excl_prompt = forward_stats.get('average_forwards_excluding_prompt', 0.0)
            total_fwds = forward_stats.get('total_forwards', 0)
            avg_active_fwd = forward_stats.get('avg_active_blocks_per_forward', 0.0)
            avg_active_step = forward_stats.get('avg_active_blocks_per_step', 0.0)
            avg_forwards_per_block = forward_stats.get('avg_forwards_per_block', 0.0)
            avg_forwarded_blocks_per_forward = forward_stats.get('avg_forwarded_blocks_per_forward', 0.0)
            # Pretty print per-block forwards sorted by block id
            sorted_items = sorted(per_block.items(), key=lambda x: x[0])
            per_block_str = ', '.join([f"{bid}:{cnt}" for bid, cnt in sorted_items])
            print(f"[stats] total_forwards={total_fwds}; avg_forwards_per_block={avg_forwards_per_block:.2f}; blockwise_avg_forwards={avg_excl_prompt:.2f}")
            print(f"[stats] per_block_forwards: {per_block_str}")
            print(f"[stats] avg_forwarded_blocks_per_forward={avg_forwarded_blocks_per_forward:.2f}; avg_active_blocks_per_forward={avg_active_fwd:.2f}; avg_active_blocks_per_step={avg_active_step:.2f}")
            decoding_order = forward_stats.get('decoding_order', None)
            trimmed_positions = []
            print(f"Decoding steps: {len(decoding_order)}, total forwards: {total_fwds}")
            if decoding_order is not None:
                print(f"[order] decoding_order has {len(decoding_order)} steps")
                # Print each step as a set of absolute positions updated at that step.
                for i, pos_list in enumerate(decoding_order):
                    if len(pos_list) == 0:
                        continue
                    print(f"[order] step {i}: {pos_list}")
                # NOTE: Debugging, print per-step decoded tokens
                # `decoding_order` contains absolute positions in the output sequence, with markers:
                # - positive int: normal decode at position `pos`
                # - positive int: normal speculative decode at position `pos`, `pos` + .5 resampled, `pos` + .2 thresholded
                # - negative int: forced diffusion decode at position `-pos`
                ids_1d = gen_ids
                if isinstance(gen_ids, torch.Tensor) and gen_ids.dim() == 2:
                    ids_1d = gen_ids[0]
                ids_list = ids_1d.tolist() if isinstance(ids_1d, torch.Tensor) else list(ids_1d)
                prompt_length = inputs.shape[1]
                for step_i, pos_list in enumerate(decoding_order):
                    if not pos_list:
                        continue
                    decoded_items = []
                    for p in pos_list:
                        # Recover absolute position from marker.
                        pos = int(p) if p > 0 else int(-p)
                        if pos-prompt_length < 0 or pos-prompt_length >= len(ids_list):
                            trimmed_positions.append(pos)
                            continue  # NOTE: trimmed after EOS so not in the output sequence, skip
                        tok_id = int(ids_list[pos-prompt_length])
                        piece = tokenizer.decode(
                            [tok_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        # decoded_items.append((pos, tok_id, piece))
                        decoded_items.append((p, tok_id, piece))
                    # decoded_items.sort(key=lambda x: x[0])
                    if decoded_items:
                        msg = ", ".join([f"{pos}: {piece!r}" for pos, tid, piece in decoded_items])
                        print(f"[decode] step {step_i} -> {msg}")
                if len(trimmed_positions) > 0:
                    print(f"\n[decode] Trimmed {len(trimmed_positions)} tokens after EOS, positions: {trimmed_positions}")
        if args.prompt == "<count>":
            from countdown_eval import evaluate_countdown, summarize_results
            res = evaluate_countdown(text, start=1000, end=1)
            print(summarize_results(res, start=1000, end=1))


if __name__ == "__main__":
    main()
