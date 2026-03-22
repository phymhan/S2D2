from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch


def _create_full_block_attention_mask(
    prompt_length: int,
    max_length: int,
    block_size: int,
    device=None,
    dtype=None,
):
    if dtype is None:
        dtype = torch.bfloat16
    # Build a (max_length, max_length) block-autoregressive allowed-attention pattern using Kronecker
    # products, then convert it to the additive attention mask convention (0 for allowed, -inf for masked).
    #
    # Semantics match the original loop implementation:
    # - Prompt tokens attend fully within prompt only.
    # - Non-prompt tokens attend to all prompt tokens, and to all tokens in their own and previous blocks.
    attn_2d = torch.full((max_length, max_length), float("-inf"), device=device, dtype=dtype)

    # Prompt -> prompt is fully allowed.
    if prompt_length > 0:
        attn_2d[:prompt_length, :prompt_length] = 0

    remaining_length = max_length - prompt_length
    if remaining_length <= 0:
        return attn_2d.unsqueeze(0).unsqueeze(0)

    # Non-prompt -> prompt is always allowed.
    if prompt_length > 0:
        attn_2d[prompt_length:, :prompt_length] = 0

    num_blocks = (remaining_length + block_size - 1) // block_size

    # Block-level lower-triangular "can attend" matrix, expanded to token-level via Kronecker with an
    # all-ones (block_size x block_size) matrix. Slice back down to handle the final partial block.
    block_allowed = torch.tril(torch.ones((num_blocks, num_blocks), device=device, dtype=torch.bool))
    block_tokens = torch.ones((block_size, block_size), device=device, dtype=torch.bool)
    gen_gen_allowed = torch.kron(block_allowed, block_tokens)[:remaining_length, :remaining_length]

    gen_gen_view = attn_2d[prompt_length:, prompt_length:]
    gen_gen_view.masked_fill_(gen_gen_allowed, 0)

    return attn_2d.unsqueeze(0).unsqueeze(0)


def _extract_attention_mask(
    full_mask: torch.Tensor, start_pos: int, input_length: int, cache_length: int
) -> torch.Tensor:
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    extracted_mask = torch.full(
        (1, 1, input_length, total_length),
        -torch.inf,
        device=full_mask.device,
        dtype=full_mask.dtype,
    )
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted_mask


def _construct_2L_attention_mask(extracted_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    device = extracted_mask.device
    dtype = extracted_mask.dtype
    if extracted_mask.ndim == 4:
        extracted_mask_2d = extracted_mask.squeeze(0).squeeze(0)
    else:
        extracted_mask_2d = extracted_mask
    num_rows = extracted_mask_2d.shape[0]
    num_cols = extracted_mask_2d.shape[1]
    # extracted_mask_2d is (L, cache_len + L)
    cache_len = int(num_cols - num_rows)
    if cache_len < 0:
        cache_len = 0

    # Number of verifier blocks in the current segment (for block-diagonal mask-half attention).
    num_blocks = (num_rows + block_size - 1) // block_size

    block_tokens = torch.ones((block_size, block_size), device=device, dtype=torch.bool)
    diag_allowed = torch.eye(num_blocks, device=device, dtype=torch.bool)

    # Keys are: [cached keys (cache_len) | first-half tokens (L) | second-half mask tokens (L)]
    attn_2d = torch.full(
        (num_rows * 2, cache_len + num_rows * 2), float("-inf"), device=device, dtype=dtype
    )

    # Top half: original extracted attention into cached keys + first-half tokens.
    attn_2d[:num_rows, :num_cols] = extracted_mask_2d

    # Bottom half: same cached-key attention, but within-segment columns are shifted by `block_size`
    # to exclude self (and future tokens) while conditioning on the draft prefix.
    if cache_len > 0:
        attn_2d[num_rows:, :cache_len] = extracted_mask_2d[:, :cache_len]
    if num_rows > block_size:
        # Old within-segment columns: [cache_len : cache_len+L)
        # New within-segment columns: [cache_len : cache_len+L-block_size)
        attn_2d[num_rows:, cache_len : cache_len + num_rows - block_size] = extracted_mask_2d[
            :, cache_len + block_size : cache_len + num_rows
        ]

    # Mask-half self attention: block-diagonal (matches the 2L trick diagram).
    mask_self = torch.full((num_rows, num_rows), float("-inf"), device=device, dtype=dtype)
    mask_self.masked_fill_(torch.kron(diag_allowed, block_tokens)[:num_rows, :num_rows], 0)
    attn_2d[num_rows:, cache_len + num_rows :] = mask_self
    return attn_2d.unsqueeze(0).unsqueeze(0)


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    *,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)

    confidence = initial_confidence.clone()
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0, initial_confidence


def shift_logits_right(logits: torch.Tensor, *, last_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Shift logits right by one position (autoregressive alignment).

    Matches `eval_dream.py`'s `_shift_logits` behavior:
    - shifted[:, 1:] = logits[:, :-1]
    - shifted[:, 0] = last_logits (if provided) else 1.0
    """
    if logits.shape[1] == 0:
        raise RuntimeError("logits sequence length is 0")

    shifted = torch.zeros_like(logits)
    shifted[:, 1:, :] = logits[:, :-1, :]
    if last_logits is not None:
        ll = last_logits
        if isinstance(ll, torch.Tensor) and ll.ndim == 3 and ll.size(1) == 1:
            ll = ll[:, 0, :]
        shifted[:, 0, :] = ll
    else:
        shifted[:, 0, :] = 1.0
    return shifted


@torch.inference_mode()
def generate_block_single(
    model,
    input_ids: torch.LongTensor,
    *,
    max_length: int,
    max_new_tokens: int,
    block_size: int,
    mask_token_id: int,
    model_type: str = "llada",
    eos_token_id: Optional[Union[int, List[int]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    block_add_threshold: float = 0.1,
    decoded_token_threshold: float = 0.5,
    skip_threshold: float = 0.9,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    shift_logits: Optional[bool] = None,
    return_forward_stats: bool = False,
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, Dict]]:
    """
    Standalone version of `LLaDAModelLM.generate_block_single`.

    Call it like: `generate_block_single(model, input_ids, ...)`.
    """
    device = device or (
        input_ids.device
        if isinstance(input_ids, torch.Tensor)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_type = (model_type or "llada").lower()
    if shift_logits is None:
        shift_logits = model_type == "dream"

    stop_token_ids: Optional[List[int]]
    if eos_token_id is None:
        stop_token_ids = None
    elif isinstance(eos_token_id, int):
        stop_token_ids = [int(eos_token_id)]
    else:
        stop_token_ids = [int(x) for x in list(eos_token_id)]
    stop_ids_tensor = (
        torch.tensor(stop_token_ids, device=device, dtype=torch.long) if stop_token_ids else None
    )

    x_t = input_ids.to(device)
    prompt_length = x_t.shape[1]

    full_attention_mask = _create_full_block_attention_mask(
        prompt_length=prompt_length,
        max_length=max_length,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )

    block_states: Dict[int, Dict] = {
        0: {
            "start_pos": 0,
            "end_pos": prompt_length,
            "mask_count": 0,
            "total_masks": prompt_length,
            "state": "to_cache",
            "is_complete": True,
        }
    }
    past_key_values = None
    current_blocks = 0
    step = 0
    eos_detected = False
    cache_length = 0
    last_logits = None

    if return_forward_stats:
        from collections import defaultdict as _dd

        block_forward_counts: Dict[int, int] = _dd(int)
        total_forwards: int = 0
        step_count: int = 0
        sum_active_blocks_over_steps: int = 0
        sum_active_blocks_over_forwards: int = 0
        sum_forwarded_blocks_over_forwards: int = 0
        # Record decoding order per outer step (list of absolute positions unmasked each step).
        decoding_order: List[List[int]] = []

    while True:
        step += 1
        # Cache length depends on the backend cache type.
        if past_key_values is None:
            cache_length = 0
        else:
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = int(past_key_values.get_seq_length())
            else:
                try:
                    # transformers-style tuple cache: (k, v) per layer
                    cache_length = int(past_key_values[0][0].size(-2))
                except Exception:
                    # fall back to the tracked value
                    cache_length = int(cache_length)

        if return_forward_stats:
            # Count current active blocks at this step
            current_active_blocks = sum(1 for st in block_states.values() if st["state"] == "active")
            sum_active_blocks_over_steps += current_active_blocks
            step_count += 1
            step_decoding_positions: List[int] = []

        if len(block_states) - 1 < (max_new_tokens // block_size) and not eos_detected:
            last_block_id = max(block_states.keys())
            progress = 1.0
            if block_states[last_block_id]["total_masks"] > 0:
                progress = (
                    block_states[last_block_id]["total_masks"] - block_states[last_block_id]["mask_count"]
                ) / block_states[last_block_id]["total_masks"]
            if progress >= block_add_threshold:
                new_block_id = last_block_id + 1
                new_start_pos = x_t.shape[1]
                if new_start_pos + block_size <= max_length:
                    x_t = torch.cat(
                        [
                            x_t,
                            torch.full(
                                (1, block_size),
                                mask_token_id,
                                device=device,
                                dtype=torch.long,
                            ),
                        ],
                        dim=1,
                    )
                    block_states[new_block_id] = {
                        "start_pos": new_start_pos,
                        "end_pos": new_start_pos + block_size,
                        "mask_count": block_size,
                        "total_masks": block_size,
                        "state": "active",
                        "is_complete": False,
                    }
                    current_blocks += 1

        for block_id in sorted(block_states.keys()):
            decoded_tokens = block_states[block_id]["total_masks"] - block_states[block_id]["mask_count"]
            if block_states[block_id]["total_masks"] > 0:
                decode_ratio = decoded_tokens / block_states[block_id]["total_masks"]
                if decode_ratio >= decoded_token_threshold:
                    if (block_id + 1) in block_states:
                        block_states[block_id + 1]["is_complete"] = True

        if (x_t == mask_token_id).sum() == 0 and current_blocks == 0:
            break

        blocks_to_cache = [bid for bid, state in block_states.items() if state["state"] == "to_cache"]
        update_kvcache = 0
        if blocks_to_cache:
            start_pos = block_states[min(blocks_to_cache)]["start_pos"]
            end_pos = block_states[max(blocks_to_cache)]["end_pos"]
            update_kvcache = end_pos - start_pos
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos
        else:
            active_blocks = [
                bid
                for bid, state in block_states.items()
                if state["state"] == "active" and state["start_pos"] >= cache_length
            ]
            if not active_blocks:
                break
            start_pos = min(block_states[bid]["start_pos"] for bid in active_blocks)
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos

        if input_seq.shape[1] == 0:
            break

        attention_mask = _extract_attention_mask(
            full_attention_mask, process_start_pos, input_seq.shape[1], cache_length
        )

        outputs = model(
            input_seq,
            attention_bias=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            update_kvcache=update_kvcache + cache_length if model_type == "llada" else update_kvcache,
        )

        logits = outputs.logits
        if shift_logits:
            logits = shift_logits_right(logits, last_logits=last_logits)

        if return_forward_stats:
            # Count this forward for all blocks that overlap the processed segment
            included_block_ids = [
                bid
                for bid, st in block_states.items()
                if (st["end_pos"] > process_start_pos) and (st["start_pos"] < x_t.shape[1])
            ]
            for bid in included_block_ids:
                block_forward_counts[bid] += 1
            total_forwards += 1
            # Active blocks included in this forward (overlapping processed segment)
            included_active_block_ids = [bid for bid in included_block_ids if block_states[bid]["state"] == "active"]
            sum_active_blocks_over_forwards += len(included_active_block_ids)
            sum_forwarded_blocks_over_forwards += int(input_seq.shape[1] / block_size)

        if update_kvcache > 0:
            if shift_logits:
                # Store logits of the last cached position for next alignment.
                cache_end_idx = update_kvcache - 1
                last_logits = outputs.logits[:, cache_end_idx, :].unsqueeze(1)
            past_key_values = outputs.past_key_values
            for bid in blocks_to_cache:
                block_states[bid]["state"] = "in_cache"

        blocks_to_deactivate = []
        for block_id, state in block_states.items():
            if state["state"] != "active":
                continue
            block_start = state["start_pos"]
            block_end = state["end_pos"]
            block_mask_locs = (x_t[0, block_start:block_end] == mask_token_id).nonzero().squeeze(-1)
            if block_mask_locs.numel() == 0:
                blocks_to_deactivate.append(block_id)
                continue
            logit_offset = block_start - process_start_pos
            block_mask_logits = logits[:, logit_offset + block_mask_locs, :]

            confidence, x0, initial_confidence = _sample_tokens(
                block_mask_logits.squeeze(0),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=margin_confidence,
                neg_entropy=neg_entropy,
            )

            high_conf_indices = (initial_confidence > skip_threshold).nonzero().squeeze(-1)
            if state["is_complete"] and high_conf_indices.numel() == 0 and block_mask_logits.numel() > 0:
                _, top_idx = torch.topk(confidence, 1)
                selected_indices = top_idx
                lower_than_threshold = True
            else:
                selected_indices = high_conf_indices
                lower_than_threshold = False

            if selected_indices.numel() > 0:
                positions_to_update = block_start + block_mask_locs[selected_indices]
                x_t[0, positions_to_update] = x0[selected_indices]
                state["mask_count"] -= selected_indices.numel()
                if stop_ids_tensor is not None and torch.isin(x0[selected_indices], stop_ids_tensor).any():
                    eos_detected = True
                if return_forward_stats:
                    if lower_than_threshold:
                        step_decoding_positions.extend([-int(p) for p in positions_to_update.tolist()])
                    else:
                        step_decoding_positions.extend([int(p) for p in positions_to_update.tolist()])
            if state["mask_count"] == 0:
                blocks_to_deactivate.append(block_id)

        for bid in blocks_to_deactivate:
            if block_states[bid]["state"] == "active" and all(
                block_states.get(i, {}).get("state") != "active" for i in range(bid)
            ):
                block_states[bid]["state"] = "to_cache"
                current_blocks -= 1
        if return_forward_stats:
            decoding_order.append(step_decoding_positions)

        if update_kvcache > 0:
            cache_length += update_kvcache

        if step > 10000:
            break

    gen_ids = x_t[0, prompt_length:]
    if stop_token_ids:
        first_stop_pos: Optional[int] = None
        for stop_id in stop_token_ids:
            stop_positions = (gen_ids == stop_id).nonzero()
            if stop_positions.numel() == 0:
                continue
            pos = int(stop_positions[0, 0].item())
            if first_stop_pos is None or pos < first_stop_pos:
                first_stop_pos = pos
        if first_stop_pos is not None:
            # Exclude the stop token itself (match typical HF generation behavior).
            gen_ids = gen_ids[:first_stop_pos]

    if return_forward_stats:
        non_prompt_counts = [count for bid, count in block_forward_counts.items() if bid != 0]
        avg_excl_prompt = float(sum(non_prompt_counts) / len(non_prompt_counts)) if len(non_prompt_counts) > 0 else 0.0
        avg_active_blocks_per_forward = float(sum_active_blocks_over_forwards) / float(max(1, total_forwards))
        avg_active_blocks_per_step = float(sum_active_blocks_over_steps) / float(max(1, step_count))
        avg_forwarded_blocks_per_forward = float(sum_forwarded_blocks_over_forwards) / float(max(1, total_forwards))
        num_non_prompt_blocks = int(len(non_prompt_counts))
        stats = {
            "per_block_forwards": {int(bid): int(cnt) for bid, cnt in sorted(block_forward_counts.items())},
            "average_forwards_excluding_prompt": avg_excl_prompt,
            "total_forwards": int(total_forwards),
            "avg_forwards_per_block": float(total_forwards) / float(max(1, num_non_prompt_blocks)),
            "avg_active_blocks_per_forward": float(avg_active_blocks_per_forward),
            "avg_active_blocks_per_step": float(avg_active_blocks_per_step),
            "avg_forwarded_blocks_per_forward": float(avg_forwarded_blocks_per_forward),
            "decoding_order": decoding_order,
        }
        return gen_ids, stats
    else:
        return gen_ids


@torch.inference_mode()
def generate_block_speculative(
    model,
    input_ids: torch.LongTensor,
    *,
    max_length: int,
    max_new_tokens: int,
    block_size: int,
    mask_token_id: int,
    model_type: str = "llada",
    eos_token_id: Optional[Union[int, List[int]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    block_add_threshold: float = 0.1,
    decoded_token_threshold: float = 0.5,
    skip_threshold: float = 0.9,
    draft_confidence_threshold: float = None,
    verify_confidence_threshold: float = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    shift_logits: Optional[bool] = None,
    return_forward_stats: bool = False,
    min_tokens_per_step: int = 1,
    target_block_size: int = 1,
    ssd_partial_ar_span: int = 0,
    ssd_threshold_draft_confidence: bool = False,
    ssd_ratio_tempering_factor: float = 1.0,
    min_ssd_span_length: int = 1,
    cache_ver: bool = False,
    draft_ver: bool = False,
    allow_resample: bool = True,
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, Dict]]:
    """
    Standalone version of `LLaDAModelLM.generate_block_speculative` (self-speculative decoding / SSD).

    Call it like: `generate_block_speculative(model, input_ids, ...)`.
    """
    assert cache_ver == draft_ver, "cache_ver and draft_ver must be both True or both False"
    device = device or (
        input_ids.device
        if isinstance(input_ids, torch.Tensor)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_type = (model_type or "llada").lower()
    if shift_logits is None:
        shift_logits = model_type == "dream"
    assert shift_logits == (model_type == "dream"), "shift_logits should be True if model_type is dream"

    stop_token_ids: Optional[List[int]]
    if eos_token_id is None:
        stop_token_ids = None
    elif isinstance(eos_token_id, int):
        stop_token_ids = [int(eos_token_id)]
    else:
        stop_token_ids = [int(x) for x in list(eos_token_id)]
    stop_ids_tensor = (
        torch.tensor(stop_token_ids, device=device, dtype=torch.long) if stop_token_ids else None
    )

    x_t = input_ids.to(device)
    prompt_length = x_t.shape[1]
    # For Dream (shift_logits=True), we carry logits across segmented forwards.
    last_logits = None
    _draft_confidence_threshold = draft_confidence_threshold if draft_confidence_threshold is not None else skip_threshold

    def _logits_to_probs(_logits: torch.Tensor) -> torch.Tensor:
        # Mirror `_sample_tokens` preprocessing so SSD uses the same effective distributions.
        if temperature > 0:
            _logits = _logits / temperature
        if top_p is not None and top_p < 1:
            _logits = _top_p_logits(_logits, top_p)
        if top_k is not None:
            _logits = _top_k_logits(_logits, top_k)
        return torch.softmax(_logits, dim=-1, dtype=torch.float32)

    def _reject_resample_from_delta(
        q_probs_row: torch.Tensor, p_probs_row: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        # Residual resampling distribution: normalize(max(0, q - p)).
        delta = torch.clamp(q_probs_row - p_probs_row, min=0.0)
        z = delta.sum()
        if float(z) <= 0.0 or torch.isnan(z):
            # Fallback: sample from q if the residual is degenerate.
            return torch.distributions.Categorical(probs=q_probs_row).sample()
        delta = delta / (z + eps)
        return torch.distributions.Categorical(probs=delta).sample()

    full_attention_mask = _create_full_block_attention_mask(
        prompt_length=prompt_length,
        max_length=max_length,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    full_attention_mask_ver = _create_full_block_attention_mask(
        prompt_length=prompt_length,
        max_length=max_length,
        block_size=target_block_size,
        device=device,
        dtype=dtype,
    )

    block_states: Dict[int, Dict] = {
        0: {
            "start_pos": 0,
            "end_pos": prompt_length,
            "mask_count": 0,
            "total_masks": prompt_length,
            "state": "to_cache",
            "fully_activated": True,
        }
    }
    past_key_values = None
    post_rope_cache = bool(getattr(getattr(model, "config", None), "post_rope_cache", False))
    current_blocks = 0
    step = 0
    eos_detected = False
    cache_length = 0

    if return_forward_stats:
        from collections import defaultdict as _dd

        block_forward_counts: Dict[int, int] = _dd(int)
        total_forwards: int = 0
        step_count: int = 0
        sum_active_blocks_over_steps: int = 0
        sum_active_blocks_over_forwards: int = 0
        sum_forwarded_blocks_over_forwards: int = 0
        # Record decoding order per outer step (list of absolute positions unmasked each step).
        decoding_order: List[List[int]] = []

    while True:
        step += 1
        if return_forward_stats:
            # Count current active blocks at this step
            current_active_blocks = sum(1 for st in block_states.values() if st["state"] == "active")
            sum_active_blocks_over_steps += current_active_blocks
            step_count += 1
            step_decoding_positions: List[Union[int, float]] = []

        if len(block_states) - 1 < (max_new_tokens // block_size) and not eos_detected:
            last_block_id = max(block_states.keys())
            progress = 1.0
            if block_states[last_block_id]["total_masks"] > 0:
                progress = (
                    block_states[last_block_id]["total_masks"] - block_states[last_block_id]["mask_count"]
                ) / block_states[last_block_id]["total_masks"]
            if progress >= block_add_threshold:
                new_block_id = last_block_id + 1
                new_start_pos = x_t.shape[1]
                if new_start_pos + block_size <= max_length:
                    x_t = torch.cat(
                        [
                            x_t,
                            torch.full(
                                (1, block_size),
                                mask_token_id,
                                device=device,
                                dtype=torch.long,
                            ),
                        ],
                        dim=1,
                    )
                    block_states[new_block_id] = {
                        "start_pos": new_start_pos,
                        "end_pos": new_start_pos + block_size,
                        "mask_count": block_size,
                        "total_masks": block_size,
                        "state": "active",
                        "fully_activated": False,
                    }
                    current_blocks += 1

        for block_id in sorted(block_states.keys()):
            decoded_tokens = block_states[block_id]["total_masks"] - block_states[block_id]["mask_count"]
            if block_states[block_id]["total_masks"] > 0:
                decode_ratio = decoded_tokens / block_states[block_id]["total_masks"]
                if decode_ratio >= decoded_token_threshold:
                    if (block_id + 1) in block_states:
                        block_states[block_id + 1]["fully_activated"] = True

        if (x_t == mask_token_id).sum() == 0 and current_blocks == 0:
            break

        blocks_to_cache = [bid for bid, state in block_states.items() if state["state"] == "to_cache"]
        update_kvcache = 0
        if blocks_to_cache:
            start_pos = block_states[min(blocks_to_cache)]["start_pos"]
            end_pos = block_states[max(blocks_to_cache)]["end_pos"]
            update_kvcache = end_pos - start_pos
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos
        else:
            active_blocks = [
                bid
                for bid, state in block_states.items()
                if state["state"] == "active" and state["start_pos"] >= cache_length
            ]
            if not active_blocks:
                break
            start_pos = min(block_states[bid]["start_pos"] for bid in active_blocks)
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos

        if input_seq.shape[1] == 0:
            break

        attention_mask = _extract_attention_mask(
            full_attention_mask, process_start_pos, input_seq.shape[1], cache_length
        )

        outputs = model(
            input_seq,
            attention_bias=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            update_kvcache=update_kvcache + cache_length if model_type == "llada" else update_kvcache,
        )

        logits = outputs.logits
        if shift_logits:
            logits = shift_logits_right(logits, last_logits=last_logits)

        if return_forward_stats:
            included_block_ids = [
                bid
                for bid, st in block_states.items()
                if (st["end_pos"] > process_start_pos) and (st["start_pos"] < x_t.shape[1])
            ]
            for bid in included_block_ids:
                block_forward_counts[bid] += 1
            total_forwards += 1
            included_active_block_ids = [bid for bid in included_block_ids if block_states[bid]["state"] == "active"]
            sum_active_blocks_over_forwards += len(included_active_block_ids)
            sum_forwarded_blocks_over_forwards += int(input_seq.shape[1] / block_size)

        if update_kvcache > 0:
            if shift_logits:
                cache_end_idx = update_kvcache - 1
                last_logits = outputs.logits[:, cache_end_idx, :].unsqueeze(1)
            past_key_values = outputs.past_key_values
            for bid in blocks_to_cache:
                block_states[bid]["state"] = "in_cache"

        # Sample draft tokens once per step for all active masked positions.
        active_mask_abs_positions: List[int] = []
        active_mask_rel_positions: List[int] = []
        active_pos_to_block: Dict[int, int] = {}
        block_mask_pos_list: Dict[int, torch.Tensor] = {}

        for block_id, state in block_states.items():
            if state["state"] != "active":
                continue
            block_start = state["start_pos"]
            block_end = state["end_pos"]
            block_mask_locs = (x_t[0, block_start:block_end] == mask_token_id).nonzero().squeeze(-1)
            if block_mask_locs.numel() == 0:
                continue
            abs_pos_t = (block_start + block_mask_locs).to(dtype=torch.long)
            block_mask_pos_list[block_id] = abs_pos_t
            for p in abs_pos_t.tolist():
                p_i = int(p)
                rel = p_i - int(process_start_pos)
                active_pos_to_block[p_i] = block_id
                active_mask_abs_positions.append(p_i)
                active_mask_rel_positions.append(rel)

        draft_token_by_pos: Dict[int, torch.Tensor] = {}
        draft_conf_by_pos: Dict[int, torch.Tensor] = {}
        draft_init_conf_by_pos: Dict[int, torch.Tensor] = {}
        draft_probs_by_pos: Dict[int, torch.Tensor] = {}
        pos_to_block: Dict[int, int] = dict(active_pos_to_block)

        if len(active_mask_abs_positions) > 0:
            rel_idx = torch.tensor(active_mask_rel_positions, device=device, dtype=torch.long)
            draft_logits_2d = logits[0, rel_idx, :]  # (M, V)
            confidence_all, x0_all, init_conf_all = _sample_tokens(
                draft_logits_2d,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=margin_confidence,
                neg_entropy=neg_entropy,
            )
            p_probs_all = _logits_to_probs(draft_logits_2d)  # (M, V)
            for j, abs_p in enumerate(active_mask_abs_positions):
                draft_token_by_pos[int(abs_p)] = x0_all[j]
                draft_conf_by_pos[int(abs_p)] = confidence_all[j]
                draft_init_conf_by_pos[int(abs_p)] = init_conf_all[j]
                draft_probs_by_pos[int(abs_p)] = p_probs_all[j]

        # Contiguous L2R verification span.
        start_pos_ver = process_start_pos + update_kvcache
        span_positions: List[int] = []
        ver_view = x_t[0, start_pos_ver:]
        first_mask_rel = (ver_view == mask_token_id).nonzero().squeeze(-1)
        if first_mask_rel.numel() > 0:
            idx = int(first_mask_rel.min().item())
            while idx < ver_view.shape[0] and int(ver_view[idx].item()) == int(mask_token_id):
                pos_i = int(start_pos_ver + idx)
                if pos_i not in draft_token_by_pos:
                    break
                span_positions.append(pos_i)
                idx += 1
        
        if len(span_positions) < int(min_ssd_span_length):
            span_positions = []
        
        update_tokens_by_pos: Dict[int, int] = {}
        resampled_positions: List[int] = []
        forced_positions: List[int] = []

        # Verifier setup: start after the current cache update.
        if len(span_positions) > 0:
            input_seq_ver = x_t[:, start_pos_ver:].clone()
            L_ver = int(input_seq_ver.shape[1])
            if L_ver > 0 and draft_token_by_pos:
                rel_pos = []
                rel_tok = []
                for abs_p, tok in draft_token_by_pos.items():
                    rp = int(abs_p) - int(start_pos_ver)
                    if 0 <= rp < L_ver:
                        rel_pos.append(rp)
                        rel_tok.append(int(tok.item()))
                if rel_pos:
                    rel_pos_t = torch.tensor(rel_pos, device=device, dtype=torch.long)
                    rel_tok_t = torch.tensor(rel_tok, device=device, dtype=torch.long)
                    input_seq_ver[0, rel_pos_t] = rel_tok_t
            assert (input_seq_ver==mask_token_id).sum() == 0, "input_seq_ver should not contain mask tokens"

            if model_type == 'llada':
                mask_tokens = torch.full_like(input_seq_ver, mask_token_id)
                input_seq_ver = torch.cat([input_seq_ver, mask_tokens], dim=1)

                attention_mask_ver_base = _extract_attention_mask(
                    full_attention_mask_ver,
                    start_pos_ver,
                    L_ver,
                    update_kvcache + cache_length,
                )
                attention_mask_ver = _construct_2L_attention_mask(attention_mask_ver_base, target_block_size)

                cache_len_expected = int(update_kvcache + cache_length)
                cur_pos = torch.arange(
                    cache_len_expected,
                    cache_len_expected + L_ver,
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                seq_pos = torch.cat([cur_pos, cur_pos], dim=1)  # (1, 2L)
                if post_rope_cache:
                    position_ids_ver = seq_pos
                else:
                    if cache_len_expected > 0:
                        past_pos = torch.arange(0, cache_len_expected, device=device, dtype=torch.long).unsqueeze(0)
                        position_ids_ver = torch.cat([past_pos, seq_pos], dim=1)
                    else:
                        position_ids_ver = seq_pos
                outputs_ver = model(
                    input_seq_ver,
                    attention_bias=attention_mask_ver,
                    position_ids=position_ids_ver,
                    past_key_values=past_key_values,
                    use_cache=False,
                    update_kvcache=0,
                )
                logits_ver = outputs_ver.logits[:, L_ver:, :]
            else:  # model_type == 'dream'
                attention_mask_ver = _extract_attention_mask(
                    full_attention_mask_ver,
                    start_pos_ver,
                    L_ver,
                    update_kvcache + cache_length,
                )
                outputs_ver = model(
                    input_seq_ver,
                    attention_bias=attention_mask_ver,
                    past_key_values=past_key_values,
                    use_cache=False,
                    update_kvcache=0,
                )
                logits_ver = shift_logits_right(outputs_ver.logits, last_logits=last_logits)
        
            if return_forward_stats:
                total_forwards += 1

            span_rel = torch.tensor([p - start_pos_ver for p in span_positions], device=device, dtype=torch.long)
            # q_logits = logits_ver[0, span_rel + L_ver, :]  # (S, V)
            q_logits = logits_ver[0, span_rel, :]  # (S, V)
            q_probs = _logits_to_probs(q_logits)  # (S, V)

            p_probs = torch.stack([draft_probs_by_pos[p] for p in span_positions], dim=0)  # (S, V)
            draft_tokens = torch.stack([draft_token_by_pos[p] for p in span_positions], dim=0).to(dtype=torch.long)  # (S,)

            p_sel = p_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            q_sel = q_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)
            # ratios = (q_sel / p_sel).clamp(max=1.0)
            if ssd_ratio_tempering_factor != 1.0:
                ratios = (q_sel / p_sel) ** ssd_ratio_tempering_factor
            else:
                ratios = q_sel / p_sel
            ratios = ratios.clamp(max=1.0)
            rand = torch.rand((ratios.shape[0],), device=device, dtype=ratios.dtype)
            accept_flags = rand < ratios
            if ssd_threshold_draft_confidence:
                draft_init_conf = torch.stack([draft_init_conf_by_pos[p] for p in span_positions], dim=0).to(
                    device=device, dtype=ratios.dtype
                )
                accept_flags = accept_flags & (draft_init_conf > float(_draft_confidence_threshold))

            reject_idx_tensor = (~accept_flags).nonzero().squeeze(-1)
            first_reject = int(reject_idx_tensor[0].item()) if reject_idx_tensor.numel() > 0 else None

            if first_reject is None:
                for i, pos_i in enumerate(span_positions):
                    update_tokens_by_pos[pos_i] = int(draft_tokens[i].item())
            else:
                for i in range(first_reject):
                    pos_i = span_positions[i]
                    update_tokens_by_pos[pos_i] = int(draft_tokens[i].item())

                rej_pos = span_positions[first_reject]
                # if ssd_threshold_draft_confidence:
                #     allow_resample = bool(float(draft_init_conf_by_pos[rej_pos].item()) > float(skip_threshold))
                if allow_resample and (verify_confidence_threshold is None or q_sel[first_reject] >= float(verify_confidence_threshold)):
                    resampled = _reject_resample_from_delta(q_probs[first_reject], p_probs[first_reject])
                    update_tokens_by_pos[rej_pos] = int(resampled.item())
                    resampled_positions.append(rej_pos)

                if ssd_partial_ar_span and ssd_partial_ar_span > 0 and step > 1:
                    i = first_reject + 1
                    S = len(span_positions)
                    while i < S:
                        if bool(accept_flags[i].item()):
                            j = i + 1
                            while j < S and bool(accept_flags[j].item()):
                                j += 1
                            run_len = j - i
                            if run_len >= int(ssd_partial_ar_span):
                                for k in range(i, j):
                                    pos_i = span_positions[k]
                                    update_tokens_by_pos[pos_i] = int(draft_tokens[k].item())
                            i = j
                        else:
                            i += 1

        # Enforce minimum decoded tokens per step for fully-activated blocks.
        updated_pos_set = set(update_tokens_by_pos.keys())
        updated_count_by_block: Dict[int, int] = {}
        for pos_i in updated_pos_set:
            bid = pos_to_block.get(pos_i, None)
            if bid is not None:
                updated_count_by_block[bid] = updated_count_by_block.get(bid, 0) + 1

        # min_k = int(max(0, int(ssd_min_tokens_per_step or 0)))
        min_k = max(1, min_tokens_per_step or 1)
        # Fallback to diffusion sampling
        for block_id, state in block_states.items():
            if state.get("state") != "active" or (not state.get("fully_activated", False)):
                continue
            already = int(updated_count_by_block.get(block_id, 0))
            if already >= min_k:
                continue
            abs_pos = block_mask_pos_list.get(block_id, None)
            if abs_pos is None or abs_pos.numel() == 0:
                continue
            remaining = [int(p.item()) for p in abs_pos if int(p.item()) not in updated_pos_set]
            if not remaining:
                continue
            need = min(min_k - already, len(remaining))
            confs = torch.stack([draft_conf_by_pos[p] for p in remaining], dim=0)
            # If enough positions are above `skip_threshold`, accept ALL of them (descending confidence);
            # otherwise, take top-k to satisfy the minimum tokens per step.
            high_conf_mask = confs >= float(skip_threshold)
            num_high_conf = int(high_conf_mask.sum().item())
            if num_high_conf >= need:
                sel_idx = high_conf_mask.nonzero(as_tuple=False).squeeze(-1)
                if sel_idx.numel() > 0:
                    sel_confs = confs[sel_idx]
                    _, order_idx = torch.sort(sel_confs, descending=True)
                    chosen_idx = sel_idx[order_idx].tolist()
                else:
                    chosen_idx = []
            else:
                _, top_idx = torch.topk(confs, k=need)
                chosen_idx = top_idx.tolist()

            for idx in chosen_idx:
                pos_i = remaining[int(idx)]
                update_tokens_by_pos[pos_i] = int(draft_token_by_pos[pos_i].item())
                forced_positions.append(pos_i)
                updated_pos_set.add(pos_i)
                updated_count_by_block[block_id] = updated_count_by_block.get(block_id, 0) + 1

        # Apply updates and update per-block state.
        if update_tokens_by_pos:
            upd_positions = torch.tensor(list(update_tokens_by_pos.keys()), device=device, dtype=torch.long)
            upd_tokens = torch.tensor(
                [update_tokens_by_pos[int(p)] for p in upd_positions.tolist()],
                device=device,
                dtype=torch.long,
            )
            x_t[0, upd_positions] = upd_tokens

            if stop_ids_tensor is not None and torch.isin(upd_tokens, stop_ids_tensor).any():
                eos_detected = True

            for pos_i in update_tokens_by_pos.keys():
                bid = pos_to_block.get(int(pos_i), None)
                if bid is None:
                    continue
                block_states[bid]["mask_count"] -= 1

            if return_forward_stats:
                for pos_i in update_tokens_by_pos.keys():
                    if int(pos_i) in resampled_positions:
                        step_decoding_positions.append(float(pos_i)+0.5)
                    elif int(pos_i) in forced_positions:
                        step_decoding_positions.append(-int(pos_i))
                    else:
                        step_decoding_positions.append(int(pos_i))

        blocks_to_deactivate: List[int] = []

        # Mark completed blocks for caching.
        for block_id, state in block_states.items():
            if state["state"] != "active":
                continue
            if state["mask_count"] == 0:
                blocks_to_deactivate.append(block_id)

        for bid in blocks_to_deactivate:
            if block_states[bid]["state"] == "active" and all(
                block_states.get(i, {}).get("state") != "active" for i in range(bid)
            ):
                block_states[bid]["state"] = "to_cache"
                current_blocks -= 1
        if return_forward_stats:
            decoding_order.append(step_decoding_positions)

        if update_kvcache > 0:
            cache_length += update_kvcache

        if step > 10000:
            break

    gen_ids = x_t[0, prompt_length:]
    if stop_token_ids:
        first_stop_pos: Optional[int] = None
        for stop_id in stop_token_ids:
            stop_positions = (gen_ids == stop_id).nonzero()
            if stop_positions.numel() == 0:
                continue
            pos = int(stop_positions[0, 0].item())
            if first_stop_pos is None or pos < first_stop_pos:
                first_stop_pos = pos
        if first_stop_pos is not None:
            gen_ids = gen_ids[:first_stop_pos]

    if return_forward_stats:
        non_prompt_counts = [count for bid, count in block_forward_counts.items() if bid != 0]
        avg_excl_prompt = float(sum(non_prompt_counts) / len(non_prompt_counts)) if len(non_prompt_counts) > 0 else 0.0
        avg_active_blocks_per_forward = float(sum_active_blocks_over_forwards) / float(max(1, total_forwards))
        avg_active_blocks_per_step = float(sum_active_blocks_over_steps) / float(max(1, step_count))
        avg_forwarded_blocks_per_forward = float(sum_forwarded_blocks_over_forwards) / float(max(1, total_forwards))
        num_non_prompt_blocks = int(len(non_prompt_counts))
        stats = {
            "per_block_forwards": {int(bid): int(cnt) for bid, cnt in sorted(block_forward_counts.items())},
            "average_forwards_excluding_prompt": avg_excl_prompt,
            "total_forwards": int(total_forwards),
            "avg_forwards_per_block": float(total_forwards) / float(max(1, num_non_prompt_blocks)),
            "avg_active_blocks_per_forward": float(avg_active_blocks_per_forward),
            "avg_active_blocks_per_step": float(avg_active_blocks_per_step),
            "avg_forwarded_blocks_per_forward": float(avg_forwarded_blocks_per_forward),
            "decoding_order": decoding_order,
        }
        return gen_ids, stats
    else:
        return gen_ids

