import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict
import random
import time


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0, do_sample=False):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if not do_sample or temperature < 1e-6:  # NOTE: treat as 0, temperature=0 is treated as =1 with do_sample=False
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # [batch*block, vocab]
        token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch*block, 1]
        token_prob = torch.gather(probs, -1, token)         # [batch*block, 1]
        return token.view(*orig_shape), token_prob.view(*orig_shape)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        input_ids,
        attention_mask=None,
        mask_id=151669, # <|MASK|>
        gen_length=128,
        max_gen_toks=None,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        return_forward_stats=False,
        do_sample=False,
        trim_stop_tokens=True,
        **kwargs,
    ):
    if return_forward_stats:
        decoding_order: List[List[int]] = []

    # lm_eval compatibility: `max_gen_toks` is equivalent to our `gen_length`.
    if max_gen_toks is not None:
        gen_length = int(max_gen_toks)

    model.eval()
    batch_size, prompt_length = input_ids.shape
    assert batch_size == 1, "batch size must be 1"
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((batch_size, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    total_forward_steps = 0

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)
        total_forward_steps += 1

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)
    block_unmasked_count = [0 if i < prefill_blocks else -1 for i in range(num_blocks)]  # NOTE: -1: not visited, >= 0: unmasked count

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        block_unmasked_count[num_block] = 0
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                total_forward_steps += 1
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits
            total_forward_steps += 1

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

            # Sampling strategy
            step_pos: List[int] = ['|'] if block_unmasked_count[num_block] == 0 else []
            transfer_positions: List[List[int]] = [[] for _ in range(cur_x.shape[0])]
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                        k = int(num_transfer_tokens[step].item())
                        transfer_positions[j] = list(range(int(first_mask_index), int(first_mask_index) + k))
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True
                    transfer_positions[j] = [int(p) for p in idx.tolist()]

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                        # Preserve an order: descending confidence among selected positions.
                        sel_pos = high_conf_mask.nonzero(as_tuple=True)[0]
                        if sel_pos.numel() > 0:
                            sel_conf = confidence[j, sel_pos]
                            _, order_idx = torch.sort(sel_conf, descending=True)
                            transfer_positions[j] = [int(sel_pos[t].item()) for t in order_idx.tolist()]
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
                        transfer_positions[j] = [int(p) for p in idx.tolist()]
            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(x0_p.shape[0]):
                    k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_token_indices = order[j, :k]
                    transfer_index[j, selected_token_indices] = True
                    transfer_positions[j] = [int(p) for p in selected_token_indices.tolist()]
                
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")
            
            block_unmasked_count[num_block] += int(transfer_index.sum().item())

            if return_forward_stats:
                # Record as +int absolute positions, preserving the selection order.
                base_abs = int(num_block * block_length)
                step_pos.extend([base_abs + int(p) for p in transfer_positions[0]])

            cur_x[transfer_index] = x0[transfer_index]
            if return_forward_stats:
                decoding_order.append(step_pos)

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    # Trim: drop any trailing mask padding, and drop stop token + everything after it.
    # (When stopping happens mid-generation, later positions are still mask_id.)
    if trim_stop_tokens and x.numel() > 0:
        gen = x[:, prompt_length:]  # [batch, gen+pad]
        if gen.numel() > 0:
            # Last non-mask position (to remove untouched mask padding).
            non_mask = (gen != mask_id)
            non_mask_idx = non_mask.nonzero(as_tuple=False)
            last_non_mask_pos = None
            if non_mask_idx.numel() > 0:
                # non_mask_idx is (N, 2): [batch_idx, pos]
                last_non_mask_pos = int(non_mask_idx[:, 1].max().item())

            # First stop position (to remove stop_id and any tokens after it).
            first_stop_pos = None
            if stopping_criteria_idx is not None:
                for stop_id in stopping_criteria_idx:
                    stop_idx = (gen == stop_id).nonzero(as_tuple=False)
                    if stop_idx.numel() == 0:
                        continue
                    pos = int(stop_idx[:, 1].min().item())
                    if first_stop_pos is None or pos < first_stop_pos:
                        first_stop_pos = pos

            # Compute final length (batch_size is asserted 1 above).
            if last_non_mask_pos is None:
                valid_gen_len = 0
            else:
                valid_gen_len = last_non_mask_pos + 1
            if first_stop_pos is not None:
                valid_gen_len = min(valid_gen_len, first_stop_pos)

            x = x[:, :prompt_length + valid_gen_len]

    if return_forward_stats:
        return x, {"decoding_order": decoding_order, "total_forward_steps": total_forward_steps}
    return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default="JetLM/SDAR-8B-Chat",
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=20480,
                        help="Maximum generation length in tokens")
    parser.add_argument("--max_gen_toks", type=int, default=512,
                        help="(lm_eval compat) If set, overrides --gen_length.")
    parser.add_argument("--block_length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],)
    parser.add_argument("--return_forward_stats", action='store_true',
                        help="Return forward statistics")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to generate")
    parser.add_argument("--no_chat", action="store_true",
                        help="If set, skip chat template and use raw prompt text")
    return parser.parse_args(), parser


if __name__ == "__main__":
    args, parser = parse_args()
    set_seed(args.seed)

    if args.max_gen_toks is not None:
        args.gen_length = int(args.max_gen_toks)

    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        # dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
        dict(role="user", content="Question: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    if args.prompt is None:
        prompt = origin_prompt
    else:
        prompt = [
            dict(role="user", content=args.prompt)
        ]

    if args.no_chat:
        messages = args.prompt if args.prompt is not None else origin_prompt[0]["content"]
    else:
        messages = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages] * args.batch_size, **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    start_time = time.perf_counter()
    output_ids = block_diffusion_generate(
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
        stopping_criteria_idx=args.stopping_criteria_idx,
        return_forward_stats=args.return_forward_stats,
    )
    end_time = time.perf_counter()

    if args.return_forward_stats:
        output_ids, stats = output_ids
        decoding_order = stats.get("decoding_order", None)
        total_forward_steps = stats.get("total_forward_steps", 0)
        prompt_length = tokens["input_ids"].shape[1]
        gen_ids = output_ids[:, prompt_length:]
        trimmed_positions = []
        print(f"Decoding steps: {len(decoding_order)}, Total forward steps: {total_forward_steps}")
        if decoding_order is not None:
            print(f"[order] decoding_order has {len(decoding_order)} steps")
            is_new_block = False
            for i, pos_list in enumerate(decoding_order):
                if not pos_list:
                    continue
                if pos_list[0] == '|':
                    is_new_block = True
                    pos_list = pos_list[1:]
                    print(f"[order] step {i}: <NB> {pos_list}")
                else:
                    print(f"[order] step {i}: {pos_list}")

            # Print per-step decoded token pieces (positions are +int only).
            ids_1d = gen_ids
            if isinstance(gen_ids, torch.Tensor) and gen_ids.dim() == 2:
                ids_1d = gen_ids[0]
            ids_list = ids_1d.tolist() if isinstance(ids_1d, torch.Tensor) else list(ids_1d)
            for step_i, pos_list in enumerate(decoding_order):
                is_new_block = False
                if not pos_list:
                    continue
                if pos_list[0] == '|':
                    is_new_block = True
                    pos_list = pos_list[1:]
                decoded_items = []
                for pos in pos_list:
                    if pos - prompt_length < 0 or pos - prompt_length >= len(ids_list):
                        trimmed_positions.append(int(pos))
                        continue
                    tok_id = int(ids_list[pos - prompt_length])
                    piece = tokenizer.decode(
                        [tok_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    decoded_items.append((int(pos), tok_id, piece))
                if decoded_items:
                    if is_new_block:
                        msg = "<NB> "+", ".join([f"{pos}: {piece!r}" for pos, tid, piece in decoded_items[1:]])
                    else:
                        msg = ", ".join([f"{pos}: {piece!r}" for pos, tid, piece in decoded_items])
                    print(f"[decode] step {step_i} -> {msg}")
            if len(trimmed_positions) > 0:
                print(f"\n[decode] Trimmed {len(trimmed_positions)} tokens after EOS, positions: {trimmed_positions}")

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '[EOS]')
    print('\n'+cleaned_text)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
