"""
Generation utilities for LLaDA2.1 models with KV cache support.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from configuration_llada2_moe import LLaDA2MoeConfig
from modeling_llada2_moe_cache import LLaDA2MoeModelLM


def load_model_and_tokenizer(model_path, dtype_str="bfloat16", device_map="auto"):
    """
    Load LLaDA2.1 model and tokenizer.

    Args:
        model_path: HuggingFace model repo or local path
        dtype_str: One of "bfloat16", "float16", "float32"
        device_map: Device mapping strategy for model loading

    Returns:
        tuple: (model, tokenizer)
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    config = LLaDA2MoeConfig.from_pretrained(model_path)
    model = LLaDA2MoeModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


@torch.no_grad()
def generate_cached(
    model,
    input_ids,
    attention_mask=None,
    temperature=0.0,
    block_length=32,
    gen_length=2048,
    max_gen_toks=None,
    top_p=None,
    top_k=None,
    eos_early_stop=False,
    threshold=0.95,
    editing_threshold=0.9,
    max_post_steps=16,
    eos_id=156892,
    mask_id=156895,
    num_to_transfer=1,
    record_decoding_order=False,
    return_stats=True,
    **kwargs,
):
    """
    Generate text using iterative masked refinement with KV cache.

    This function implements block-wise generation with:
    - Prefill stage: process complete prompt blocks and store KV cache
    - Decode stage: iteratively refine each block with mask-filling and editing

    Args:
        model: LLaDA2MoeModelLM instance
        input_ids: Input token IDs (tensor)
        attention_mask: Attention mask (currently unused, reserved for future)
        temperature: Sampling temperature (0.0 for greedy)
        block_length: Tokens per generation block
        gen_length: Total number of tokens to generate
        max_gen_toks: If not None, overrides gen_length
        top_p: Nucleus sampling threshold (None to disable)
        top_k: Top-k sampling cutoff (None to disable)
        eos_early_stop: Stop generation at first EOS token
        threshold: Confidence threshold for unmasking tokens
        editing_threshold: Confidence threshold for editing non-masked tokens
        max_post_steps: Maximum global editing steps after all masks resolved
        eos_id: EOS token ID
        mask_id: Mask token ID for refinement
        num_to_transfer: Minimum masked positions to resolve per iteration
        record_decoding_order: If True, also record per-step decoding_order
        return_stats: If True, return (tensor, stats_dict); if False, return tensor only

    Returns:
        If return_stats is True (default): (tensor, stats_dict) where tensor is
        input + generated token IDs (prompt included) and stats_dict contains "nfe".
        When record_decoding_order is True, stats_dict also contains "decoding_order".
        If return_stats is False: tensor only.
    """
    if max_gen_toks is not None:
        gen_length = max_gen_toks
    input_ids = input_ids.to(model.device)

    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block-causal attention mask (for prefill stage only)
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(model.dtype)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    past_key_values = DynamicCache()
    nfe = 0
    decoding_order = [] if record_decoding_order else None

    # ── Prefill stage: process all complete prompt blocks and store KV ──
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, :prefill_length, :prefill_length
        ]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )
        nfe += 1

    # ── Decode stage: process each block with KV cache ──
    for num_block in range(prefill_blocks, num_blocks):
        block_start_pos = num_block * block_length
        block_end_pos = (num_block + 1) * block_length

        cur_x = x[:, block_start_pos:block_end_pos].clone()
        cur_position_ids = position_ids[:, block_start_pos:block_end_pos]

        # Prompt tokens within this block (if any)
        prompt_mask_in_block = torch.zeros(
            block_length, dtype=torch.bool, device=model.device
        )
        if block_start_pos < prompt_length:
            prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
            prompt_mask_in_block[:prompt_end_in_block] = True

        post_steps = 0
        while True:
            old_block_tokens = cur_x.clone()
            active_block_mask = cur_x == mask_id
            if not torch.any(active_block_mask):
                post_steps += 1
            if post_steps > max_post_steps:
                break

            step_info = {"edit": [], "unmask": []} if record_decoding_order else None

            # Forward with cache (don't store KV during denoising iterations)
            # attention_mask=None means full attention to all cached + current tokens
            outputs = model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            )
            logits = outputs.logits  # (1, block_length, vocab_size)
            nfe += 1

            x0, x0_p = model._sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # ── Mask-filling: select which masked positions to unmask ──
            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            if active_block_mask.sum() > 0:
                mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = (
                    mask_confidence[0] > threshold
                ) & active_block_mask[0]
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    mask_transfer_index[0] = high_conf_mask
                else:
                    num_available = active_block_mask.sum().item()
                    if num_available > 0:
                        _, idx = torch.topk(
                            mask_confidence[0],
                            k=min(num_to_transfer, num_available),
                        )
                        mask_transfer_index[0, idx] = True

            # ── Editing: optionally replace non-mask, non-prompt tokens ──
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            non_mask_positions = ~active_block_mask
            non_prompt_positions = ~prompt_mask_in_block
            editable_positions = non_mask_positions & non_prompt_positions[None, :]
            editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
            high_conf_editing = (
                editing_confidence[0] > editing_threshold
            ) & editable_positions[0]

            token_changed = x0[0] != old_block_tokens[0]
            editing_transfer_index[0] = high_conf_editing & token_changed
            final_transfer_index = mask_transfer_index | editing_transfer_index

            if final_transfer_index.any():
                cur_x[final_transfer_index] = x0[final_transfer_index]

            if record_decoding_order:
                base_abs = block_start_pos
                # Record mask-filling positions
                for pos in mask_transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                    abs_pos = base_abs + pos
                    step_info["unmask"].append(int(abs_pos))
                # Record editing positions
                for pos in editing_transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                    abs_pos = base_abs + pos
                    old_tok = int(old_block_tokens[0, pos].item())
                    new_tok = int(x0[0, pos].item())
                    step_info["edit"].append((abs_pos, old_tok, new_tok))
                decoding_order.append(step_info)

            if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                break

        x[:, block_start_pos:block_end_pos] = cur_x

        # Check if we should stop before committing KV (commit is only needed
        # if there are subsequent blocks that will read from the cache).
        if eos_early_stop:
            generated_part = x[0, prompt_length:block_end_pos]
            if (generated_part == mask_id).sum() == 0:
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    break

        if num_block < num_blocks - 1:
            # ── Commit finalized block: store KV for this block in cache ──
            model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )
            nfe += 1

    # ── Extract generated tokens ──
    generated_answer = x[:, : prompt_length + gen_length]
    eos_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
        as_tuple=True
    )[0]
    if len(eos_positions) > 0:
        first_eos_position = eos_positions[0].item()
    else:
        first_eos_position = gen_length

    result = generated_answer[
        :, : input_ids.shape[1] + first_eos_position + 1
    ]

    if not return_stats:
        return result
    stats = {"nfe": nfe}
    if record_decoding_order:
        stats["decoding_order"] = decoding_order
    return result, stats


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask=None,
    temperature=0.0,
    block_length=32,
    gen_length=2048,
    max_gen_toks=None,
    top_p=None,
    top_k=None,
    eos_early_stop=False,
    threshold=0.95,
    editing_threshold=0.9,
    max_post_steps=16,
    eos_id=156892,
    mask_id=156895,
    num_to_transfer=1,
    record_decoding_order=False,
    return_stats=True,
    **kwargs,
):
    """
    Generate text using iterative masked refinement without KV cache.

    Each iteration passes the full sequence x[:, :current_window_end] with
    block-causal attention. Logits are sliced to the last block_length positions.

    Args:
        model: LLaDA2MoeModelLM instance (or any compatible model)
        input_ids: Input token IDs (tensor)
        attention_mask: Attention mask (currently unused, reserved for future)
        temperature: Sampling temperature (0.0 for greedy)
        block_length: Tokens per generation block
        gen_length: Total number of tokens to generate
        max_gen_toks: If not None, overrides gen_length
        top_p: Nucleus sampling threshold (None to disable)
        top_k: Top-k sampling cutoff (None to disable)
        eos_early_stop: Stop generation at first EOS token
        threshold: Confidence threshold for unmasking tokens
        editing_threshold: Confidence threshold for editing non-masked tokens
        max_post_steps: Maximum global editing steps after all masks resolved
        eos_id: EOS token ID
        mask_id: Mask token ID for refinement
        num_to_transfer: Minimum masked positions to resolve per iteration
        record_decoding_order: If True, also record per-step decoding_order
        return_stats: If True, return (tensor, stats_dict); if False, return tensor only

    Returns:
        If return_stats is True (default): (tensor, stats_dict) where tensor is
        input + generated token IDs (prompt included) and stats_dict contains "nfe".
        When record_decoding_order is True, stats_dict also contains "decoding_order".
        If return_stats is False: tensor only.
    """
    if max_gen_toks is not None:
        gen_length = max_gen_toks
    input_ids = input_ids.to(model.device)

    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(model.dtype)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    nfe = 0
    decoding_order = [] if record_decoding_order else None

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length
        cur_x = x[:, :current_window_end]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, :current_window_end, :current_window_end
        ]
        cur_position_ids = position_ids[:, :current_window_end]
        block_start_pos = num_block * block_length

        # Prompt tokens within this block (if any)
        prompt_mask_in_block = torch.zeros(
            block_length, dtype=torch.bool, device=model.device
        )
        if block_start_pos < prompt_length:
            prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
            prompt_mask_in_block[:prompt_end_in_block] = True

        post_steps = 0
        while True:
            old_block_tokens = cur_x[:, -block_length:].clone()
            active_block_mask = cur_x[:, -block_length:] == mask_id
            if not torch.any(active_block_mask):
                post_steps += 1
            if post_steps > max_post_steps:
                break

            step_info = {"edit": [], "unmask": []} if record_decoding_order else None

            outputs = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
            )
            logits = outputs.logits
            nfe += 1

            active_logits = logits[:, -block_length:, :]
            x0, x0_p = model._sample_with_temperature_topk_topp(
                active_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # ── Mask-filling: select which masked positions to unmask ──
            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            if active_block_mask.sum() > 0:
                mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = (
                    mask_confidence[0] > threshold
                ) & active_block_mask[0]
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    mask_transfer_index[0] = high_conf_mask
                else:
                    num_available = active_block_mask.sum().item()
                    if num_available > 0:
                        _, idx = torch.topk(
                            mask_confidence[0],
                            k=min(num_to_transfer, num_available),
                        )
                        mask_transfer_index[0, idx] = True

            # ── Editing: optionally replace non-mask, non-prompt tokens ──
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            non_mask_positions = ~active_block_mask
            non_prompt_positions = ~prompt_mask_in_block
            editable_positions = non_mask_positions & non_prompt_positions[None, :]
            editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
            high_conf_editing = (
                editing_confidence[0] > editing_threshold
            ) & editable_positions[0]

            token_changed = x0[0] != old_block_tokens[0]
            editing_transfer_index[0] = high_conf_editing & token_changed
            final_transfer_index = mask_transfer_index | editing_transfer_index

            if final_transfer_index.any():
                cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

            if record_decoding_order:
                base_abs = block_start_pos
                # Record mask-filling positions
                for pos in mask_transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                    abs_pos = base_abs + pos
                    step_info["unmask"].append(int(abs_pos))
                # Record editing positions
                for pos in editing_transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                    abs_pos = base_abs + pos
                    old_tok = int(old_block_tokens[0, pos].item())
                    new_tok = int(x0[0, pos].item())
                    step_info["edit"].append((abs_pos, old_tok, new_tok))
                decoding_order.append(step_info)

            if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                break

        x[:, :current_window_end] = cur_x
        if eos_early_stop:
            generated_part = x[0, prompt_length:current_window_end]
            if (generated_part == mask_id).sum() == 0:
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    break

    # ── Extract generated tokens ──
    generated_answer = x[:, : prompt_length + gen_length]
    eos_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
        as_tuple=True
    )[0]
    if len(eos_positions) > 0:
        first_eos_position = eos_positions[0].item()
    else:
        first_eos_position = gen_length

    result = generated_answer[
        :, : input_ids.shape[1] + first_eos_position + 1
    ]

    if not return_stats:
        return result
    stats = {"nfe": nfe}
    if record_decoding_order:
        stats["decoding_order"] = decoding_order
    return result, stats


# ── SSD (Self-Speculative Decoding) helpers ──


def _get_num_transfer_tokens(block_length, steps):
    """Compute per-step transfer token schedule."""
    if steps == 0:
        return torch.tensor([], dtype=torch.int64)
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def _find_mask_spans_1d(mask_1d):
    """Find contiguous spans of True in a 1D boolean tensor. Returns list of lists of indices."""
    spans = []
    i = 0
    L = int(mask_1d.shape[0])
    while i < L:
        if not bool(mask_1d[i].item()):
            i += 1
            continue
        j = i
        while j < L and bool(mask_1d[j].item()):
            j += 1
        spans.append(list(range(i, j)))
        i = j
    return spans


def _construct_2l_verifier_attention_mask(L, cache_len, device, dtype):
    """
    Build (1, 1, 2L, cache_len+2L) float attention mask for the 2L verifier trick.

    Keys layout: [cached (cache_len) | first-half draft (L) | second-half masks (L)]
    Queries:     [first-half (L)     | second-half (L)]

    Returns float mask in SDPA format (0.0 = attend, -inf = block).
    """
    bool_mask = torch.zeros(2 * L, cache_len + 2 * L, device=device, dtype=torch.bool)
    # All queries attend to cached keys
    if cache_len > 0:
        bool_mask[:, :cache_len] = True
    # First half: token-causal within first-half
    bool_mask[:L, cache_len:cache_len + L] = torch.tril(
        torch.ones(L, L, device=device, dtype=torch.bool)
    )
    # Second half: strict-causal into first-half (exclude same-position draft token)
    bool_mask[L:, cache_len:cache_len + L] = torch.tril(
        torch.ones(L, L, device=device, dtype=torch.bool), diagonal=-1
    )
    # Second half: block-diagonal self-attention (identity)
    bool_mask[L:, cache_len + L:cache_len + 2 * L] = torch.eye(
        L, device=device, dtype=torch.bool
    )
    # Convert to float SDPA format: (1, 1, 2L, cache_len+2L)
    float_mask = torch.zeros(1, 1, 2 * L, cache_len + 2 * L, device=device, dtype=dtype)
    float_mask.masked_fill_(~bool_mask.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
    return float_mask


def _probs_from_logits(logits_2d, temperature, top_k, top_p):
    """Convert logits -> probs using sampling transforms. logits_2d: (N, V) -> (N, V)."""
    if temperature is not None and temperature > 1e-6 and temperature != 1.0:
        logits_2d = logits_2d / temperature
    if top_k is not None and top_k > 0:
        logits_2d = LLaDA2MoeModelLM._top_k_logits(logits_2d, top_k)
    if top_p is not None and top_p < 1.0:
        logits_2d = LLaDA2MoeModelLM._top_p_logits(logits_2d, top_p)
    return F.softmax(logits_2d, dim=-1, dtype=torch.float32)


def _reject_resample_from_delta(q_probs_1d, p_probs_1d):
    """
    Residual resampling: sample from normalized (q - p)+.
    Falls back to sampling from q if the residual mass is ~0.
    """
    delta = (q_probs_1d - p_probs_1d).clamp_min(0.0)
    z = float(delta.sum().item())
    if z <= 0.0 or not torch.isfinite(delta).all():
        return torch.multinomial(
            q_probs_1d.clamp_min(0).to(dtype=torch.float32), num_samples=1
        ).squeeze(0)
    delta = (delta / z).to(dtype=torch.float32)
    return torch.multinomial(delta, num_samples=1).squeeze(0)


def _estimate_token_acceptance_probs(
    estimator,
    sampled_token_confidence,
    span_logits,
    temperature,
    top_k,
    top_p,
    ssd_confidence_margin_threshold=0.05,
    ssd_entropy_temperature=1.0,
):
    """
    Estimate per-token acceptance probability alpha_i for SSD.

    Args:
        estimator: One of "hard_margin_threshold", "soft_entropy_negexp", "soft_renyi_2_entropy"
        sampled_token_confidence: (S,) confidence of sampled tokens
        span_logits: (S, V) logits at span positions
        temperature, top_k, top_p: sampling transforms for computing probs
        ssd_confidence_margin_threshold: threshold for hard_margin_threshold
        ssd_entropy_temperature: temperature for soft_entropy_negexp

    Returns:
        (S,) tensor of estimated acceptance probabilities
    """
    probs = _probs_from_logits(
        span_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )  # (S, V)

    if estimator == "hard_margin_threshold":
        if probs.shape[-1] >= 2:
            top2_vals = torch.topk(probs, k=2, dim=-1).values
            confidence_margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            confidence_margin = probs[:, 0]
        return (confidence_margin > ssd_confidence_margin_threshold).to(
            dtype=sampled_token_confidence.dtype
        )

    if estimator == "soft_entropy_negexp":
        if probs.shape[-1] <= 1:
            normalized_entropy = torch.zeros(
                probs.shape[0], device=probs.device, dtype=probs.dtype
            )
        else:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            normalized_entropy = entropy / torch.log(
                torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype)
            )
        return torch.exp(-ssd_entropy_temperature * normalized_entropy)

    if estimator == "soft_renyi_2_entropy":
        return (probs ** 2).sum(dim=-1).clamp(0.0, 1.0)

    raise ValueError(f"Unknown token acceptance estimator: {estimator}")


def _estimate_expected_accepted_tokens(alpha):
    """E[K] = sum_{k=1}^{L} prod_{i=1}^{k} alpha_i for a token span."""
    if alpha.numel() == 0:
        return 0.0
    alpha = alpha.clamp(0.0, 1.0)
    return float(torch.cumprod(alpha, dim=0).sum().item())


def _compute_do_verify_score(
    score_type,
    span_logits,
    sampled_token_confidence,
    mask_index,
    x0_p,
    threshold,
    token_acceptance_estimator,
    temperature,
    top_k,
    top_p,
    ssd_confidence_margin_threshold,
    ssd_entropy_temperature,
    score_penalty_coef,
):
    """
    Compute score for score-based verify policies.

    score_type "difference_dynamic": E[K] - c * num_high_confidence
    score_type "difference_static":  E[K] - c
    """
    span_alpha = _estimate_token_acceptance_probs(
        estimator=token_acceptance_estimator,
        sampled_token_confidence=sampled_token_confidence,
        span_logits=span_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
        ssd_entropy_temperature=ssd_entropy_temperature,
    )
    expected_accepted_tokens = _estimate_expected_accepted_tokens(span_alpha)
    c = float(score_penalty_coef)

    if score_type == "difference_dynamic":
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        high_conf_mask = confidence[0] > threshold
        num_high_confidence = int(high_conf_mask.sum().item())
        return expected_accepted_tokens - c * num_high_confidence

    if score_type == "difference_static":
        return expected_accepted_tokens - c

    raise ValueError(f"Unknown do_verify_score_type: {score_type}")


@torch.no_grad()
def generate_ssd_policy(
    model,
    input_ids,
    attention_mask=None,
    temperature=0.0,
    block_length=32,
    gen_length=2048,
    max_gen_toks=None,
    top_p=None,
    top_k=None,
    eos_early_stop=False,
    eos_id=156892,
    mask_id=156895,
    threshold=0.95,
    editing_threshold=0.9,
    min_ssd_span_length=1,
    legacy_ssd_span_strategy=False,
    ssd_ratio_tempering_factor=1.0,
    record_decoding_order=False,
    return_stats=True,
    # Policy selection
    do_verify_policy="mask_span_length",
    # Score-based policy parameters
    do_verify_score_threshold=0.0,
    hysteresis_threshold_on=0.0,
    hysteresis_threshold_off=-1.0,
    do_verify_score_type="difference_dynamic",
    score_penalty_coef=2.0,
    # Token acceptance estimator parameters
    token_acceptance_estimator="hard_margin_threshold",
    ssd_confidence_margin_threshold=0.05,
    ssd_entropy_temperature=1.0,
    num_to_transfer=1,
    max_post_steps=16,
    # minimal_topk=1,
    **kwargs,
):
    """
    Generate text using Self-Speculative Decoding (SSD) with KV cache.

    Uses the 2L verifier trick: after drafting tokens via block-diffusion sampling,
    a verification pass with a non-square attention mask performs rejection sampling
    to accept or correct drafted tokens.

    Each step also performs editing (replacing non-mask, non-prompt tokens that
    changed with high confidence), matching generate_cached behavior. This means
    when verification is always skipped (e.g. min_ssd_span_length is very large),
    the fallback is equivalent to generate_cached's mask-filling + editing.

    Verification policies:
    - mask_span_length: skip verification when the first contiguous mask span
      is shorter than min_ssd_span_length
    - score_threshold: skip verification when the expected-acceptance score
      falls below do_verify_score_threshold
    - score_hysteresis: like score_threshold but with hysteresis (on/off thresholds)

    Args:
        model: LLaDA2MoeModelLM instance
        input_ids: Input token IDs (tensor)
        attention_mask: Attention mask (currently unused, reserved for future)
        temperature: Sampling temperature (0.0 for greedy)
        block_length: Tokens per generation block
        gen_length: Total number of tokens to generate
        max_gen_toks: If not None, overrides gen_length
        top_p: Nucleus sampling threshold (None to disable)
        top_k: Top-k sampling cutoff (None to disable)
        eos_early_stop: Stop generation at first EOS token
        eos_id: EOS token ID
        mask_id: Mask token ID
        threshold: Confidence threshold for unmasking tokens (mask-filling)
        editing_threshold: Confidence threshold for editing non-masked tokens
        min_ssd_span_length: Minimum mask span length to trigger 2L verification
        legacy_ssd_span_strategy: If True, mask_span_length policy also requires
            enough high-confidence tokens before skipping verification
        ssd_ratio_tempering_factor: Exponent applied to acceptance ratios
        record_decoding_order: If True, also record per-step decoding_order
        return_stats: If True, return (tensor, stats_dict); if False, return tensor only
        do_verify_policy: Policy for deciding whether to run the 2L verifier
        do_verify_score_threshold: Threshold for score_threshold policy
        hysteresis_threshold_on: Turn-on threshold for score_hysteresis policy
        hysteresis_threshold_off: Turn-off threshold for score_hysteresis policy
        do_verify_score_type: Score function ("difference_dynamic" or "difference_static")
        score_penalty_coef: Penalty coefficient c in score computation
        token_acceptance_estimator: Estimator for per-token acceptance probability
        ssd_confidence_margin_threshold: Margin threshold for hard_margin_threshold
        ssd_entropy_temperature: Temperature for soft_entropy_negexp

    Returns:
        If return_stats is True (default): (tensor, stats_dict) where tensor is
        input + generated token IDs (prompt included) and stats_dict contains "nfe".
        When record_decoding_order is True, stats_dict also contains "decoding_order".
        If return_stats is False: tensor only.
    """
    if max_gen_toks is not None:
        gen_length = max_gen_toks
    input_ids = input_ids.to(model.device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block-causal attention mask (for prefill stage only)
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(model.dtype)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    past_key_values = DynamicCache()
    nfe = 0
    decoding_order = [] if record_decoding_order else None

    hysteresis_state = None
    if do_verify_policy == "score_hysteresis":
        hysteresis_state = {"is_on": False}

    # ── Prefill stage ──
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )
        nfe += 1

    # ── Decode stage ──
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        cur_x = x[:, block_start:block_end].clone()
        cur_position_ids = position_ids[:, block_start:block_end]

        # Prompt tokens within this block (if any)
        prompt_mask_in_block = torch.zeros(
            block_length, dtype=torch.bool, device=model.device
        )
        if block_start < prompt_length:
            prompt_end_in_block = min(prompt_length - block_start, block_length)
            prompt_mask_in_block[:prompt_end_in_block] = True

        post_steps = 0
        while True:
            old_block_tokens = cur_x.clone()
            mask_index = (cur_x == mask_id)
            if not torch.any(mask_index):
                post_steps += 1
            if post_steps > max_post_steps:
                break

            step_info = {"edit": [], "unmask": []} if record_decoding_order else None

            # ── Draft forward ──
            logits = model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
            nfe += 1

            x0, x0_p = model._sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            min_k = min(num_to_transfer, int(mask_index.sum().item()))
            decoded_this_step = 0
            transfer_index = torch.zeros_like(cur_x, dtype=torch.bool)
            update_tokens = cur_x.clone()

            # ── Editing: replace non-mask, non-prompt tokens (before SSD) ──
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            non_mask_positions = ~mask_index
            non_prompt_positions = ~prompt_mask_in_block
            editable_positions = non_mask_positions & non_prompt_positions[None, :]
            editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
            high_conf_editing = (
                editing_confidence[0] > editing_threshold
            ) & editable_positions[0]
            token_changed = x0[0] != old_block_tokens[0]
            editing_transfer_index[0] = high_conf_editing & token_changed

            if editing_transfer_index.any():
                transfer_index |= editing_transfer_index
                update_tokens[editing_transfer_index] = x0[editing_transfer_index]

                if record_decoding_order:
                    base_abs = block_start
                    for pos in editing_transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                        abs_pos = base_abs + pos
                        old_tok = int(old_block_tokens[0, pos].item())
                        new_tok = int(x0[0, pos].item())
                        step_info["edit"].append((abs_pos, old_tok, new_tok))

            # Draft-fill all masks (and include edits) for verifier first half
            input_seq_ver_first = cur_x.clone()
            if editing_transfer_index.any():
                input_seq_ver_first[editing_transfer_index] = x0[editing_transfer_index]
            if mask_index.any():
                input_seq_ver_first[mask_index] = x0[mask_index]

            # Find first contiguous mask span
            spans_rel = _find_mask_spans_1d(mask_index[0])
            span_rel = spans_rel[0] if len(spans_rel) > 0 else []
            span_rel_t = torch.tensor(span_rel, device=cur_x.device, dtype=torch.long)

            # ── Policy: decide whether to verify ──
            do_verify = True
            if do_verify_policy == "mask_span_length":
                if len(span_rel) < min_ssd_span_length:
                    if not legacy_ssd_span_strategy:
                        do_verify = False
                    else:
                        # Legacy: also require enough high-confidence tokens
                        confidence = torch.where(mask_index, x0_p, -torch.inf)
                        high_conf_count = int((confidence[0] > threshold).sum().item())
                        if high_conf_count >= min(min_ssd_span_length, int(mask_index.sum().item())):
                            do_verify = False
            elif do_verify_policy in ("score_threshold", "score_hysteresis"):
                do_verify_score = float("-inf")
                if span_rel_t.numel() > 0:
                    span_logits = logits[0].index_select(0, span_rel_t)  # (S, V)
                    do_verify_score = _compute_do_verify_score(
                        score_type=do_verify_score_type,
                        span_logits=span_logits,
                        sampled_token_confidence=x0_p[0, span_rel_t].clamp(0.0, 1.0),
                        mask_index=mask_index,
                        x0_p=x0_p,
                        threshold=threshold,
                        token_acceptance_estimator=token_acceptance_estimator,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
                        ssd_entropy_temperature=ssd_entropy_temperature,
                        score_penalty_coef=score_penalty_coef,
                    )
                if do_verify_policy == "score_threshold":
                    if do_verify_score < do_verify_score_threshold:
                        do_verify = False
                else:  # score_hysteresis
                    is_on = hysteresis_state["is_on"]
                    if is_on:
                        do_verify = not (do_verify_score < hysteresis_threshold_off)
                    else:
                        do_verify = bool(do_verify_score > hysteresis_threshold_on)
                    hysteresis_state["is_on"] = bool(do_verify)
            else:
                raise ValueError(f"Unknown do_verify_policy: {do_verify_policy}")

            if do_verify and len(span_rel) > 0:
                L = int(cur_x.shape[1])
                cache_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0

                # 2L verifier input: [draft-filled | all-mask]
                mask_tokens = torch.full_like(input_seq_ver_first, mask_id)
                input_seq_ver = torch.cat([input_seq_ver_first, mask_tokens], dim=1)  # (1, 2L)

                verify_attn_mask = _construct_2l_verifier_attention_mask(
                    L=L, cache_len=cache_len, device=cur_x.device, dtype=model.dtype
                )  # (1, 1, 2L, cache_len+2L)
                verify_position_ids = torch.cat([cur_position_ids, cur_position_ids], dim=1)  # (1, 2L)

                verify_logits = model(
                    input_seq_ver,
                    attention_mask=verify_attn_mask,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits
                nfe += 1

                # Rejection sampling over the first mask span
                q_logits = verify_logits[0, span_rel_t + L, :]  # (S, V) from masked half
                q_probs = _probs_from_logits(q_logits, temperature=temperature, top_k=top_k, top_p=top_p)

                draft_tokens = x0[0, span_rel_t]  # (S,)
                p_sel = x0_p[0, span_rel_t].clamp_min(1e-12)  # (S,)
                q_sel = q_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)  # (S,)

                if ssd_ratio_tempering_factor != 1.0:
                    ratios = (q_sel / p_sel) ** ssd_ratio_tempering_factor
                else:
                    ratios = q_sel / p_sel
                ratios = ratios.clamp(max=1.0)

                accept_flags = torch.rand(ratios.shape[0], device=ratios.device, dtype=ratios.dtype) < ratios
                reject_idx = (~accept_flags).nonzero(as_tuple=True)[0]
                first_reject = int(reject_idx[0].item()) if reject_idx.numel() > 0 else None

                # Build update list: accepted prefix + optionally resampled rejection
                update_rel = []
                if first_reject is None:
                    update_rel = list(span_rel)
                else:
                    if first_reject > 0:
                        update_rel.extend(span_rel[:first_reject])
                    # Always unmask the first rejected position (with resampled token)
                    update_rel.append(span_rel[first_reject])

                if len(update_rel) > 0:
                    upd_rel_t = torch.tensor(update_rel, device=cur_x.device, dtype=torch.long)
                    transfer_index[0, upd_rel_t] = True
                    update_tokens[0, upd_rel_t] = x0[0, upd_rel_t]

                    # Residual-resample at first rejected position
                    if first_reject is not None:
                        rej_rel = span_rel[first_reject]
                        rej_q = q_probs[first_reject]  # (V,)
                        rej_p = _probs_from_logits(
                            logits[0, rej_rel, :].unsqueeze(0),
                            temperature=temperature, top_k=top_k, top_p=top_p,
                        )[0]
                        resampled = _reject_resample_from_delta(rej_q, rej_p)
                        update_tokens[0, rej_rel] = resampled.to(dtype=torch.long)

                    decoded_this_step += len(update_rel)

                    if record_decoding_order:
                        base_abs = block_start
                        for r in update_rel:
                            abs_pos = base_abs + int(r)
                            if first_reject is not None and r == span_rel[first_reject]:
                                step_info["unmask"].append(-float(abs_pos) - 0.5)
                            else:
                                step_info["unmask"].append(-int(abs_pos))

            # ── Fallback mask-filling: if SSD didn't decode enough ──
            if decoded_this_step < min_k:
                k_left = min_k - decoded_this_step
                remaining_mask = mask_index & (~transfer_index)

                confidence = torch.where(remaining_mask, x0_p, -torch.inf)
                fallback_index = torch.zeros_like(x0, dtype=torch.bool)
                k = min(k_left, int(remaining_mask.sum().item()))
                if k > 0:
                    high_conf_mask = confidence[0] > threshold
                    num_high_confidence = int(high_conf_mask.sum().item())
                    if num_high_confidence >= k:
                        fallback_index[0] = high_conf_mask
                    else:
                        _, idx = torch.topk(confidence[0], k)
                        fallback_index[0, idx] = True

                transfer_index |= fallback_index
                update_tokens[fallback_index] = x0[fallback_index]
                decoded_this_step += int(fallback_index.sum().item())

                if record_decoding_order:
                    base_abs = block_start
                    for r in fallback_index[0].nonzero(as_tuple=True)[0].tolist():
                        step_info["unmask"].append(int(base_abs + r))

            # Apply updates
            cur_x[transfer_index] = update_tokens[transfer_index]

            if record_decoding_order:
                decoding_order.append(step_info)

            if mask_index.sum() == 0 and not editing_transfer_index.any():
                break

        x[:, block_start:block_end] = cur_x

        # Early stopping
        if eos_early_stop:
            generated_part = x[0, prompt_length:block_end]
            if (generated_part == mask_id).sum() == 0:
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    break

        # Commit KV for next block
        if num_block < num_blocks - 1:
            model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )
            nfe += 1

    # ── Extract generated tokens ──
    generated_answer = x[:, :prompt_length + gen_length]
    eos_positions = (generated_answer[0][input_ids.shape[1]:] == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        first_eos_position = eos_positions[0].item()
    else:
        first_eos_position = gen_length

    result = generated_answer[:, :input_ids.shape[1] + first_eos_position + 1]

    if not return_stats:
        return result
    stats = {"nfe": nfe}
    if record_decoding_order:
        stats["decoding_order"] = decoding_order
    return result, stats
