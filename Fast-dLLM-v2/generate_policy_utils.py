"""
Standalone generation utilities for Fast-dLLM v2.

This file extracts the model's custom `generate()` logic (implemented in the
remote-code `modeling.py`) into a standalone function, similar in spirit to
`reference/example_d2f.py` importing generation helpers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.generation.utils import GenerateDecoderOnlyOutput
import math

def _get_device(model, input_ids: torch.Tensor) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if input_ids is not None:
        return input_ids.device
    return next(model.parameters()).device


def _shift_logits_right(logits: torch.Tensor) -> torch.Tensor:
    """
    Fast-dLLM v2 inference uses right-shifted logits (predict token at position t
    from hidden state at t-1). The remote-code implementation does:
    `cat([logits[:, :1], logits[:, :-1]])`.
    """
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def _probs_from_logits(logits_2d: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    """
    Convert logits -> probs using the same effective distribution transform as draft sampling.

    logits_2d: (N, V)
    returns: (N, V) float32 probs
    """
    if temperature and float(temperature) > 0:
        logits_2d = logits_2d / float(temperature)
    probs = torch.softmax(logits_2d, dim=-1, dtype=torch.float32)
    if top_p is None or float(top_p) >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > float(top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0)
    probs_sum = torch.sum(probs, dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / probs_sum


def _reject_resample_from_delta(q_probs_1d: torch.Tensor, p_probs_1d: torch.Tensor) -> torch.Tensor:
    """
    Residual resampling used in SSD: sample from normalized (q - p)+.
    Falls back to sampling from q if the residual mass is ~0.
    """
    delta = (q_probs_1d - p_probs_1d).clamp_min(0.0)
    z = float(delta.sum().item())
    if z <= 0.0 or not torch.isfinite(delta).all():
        return torch.multinomial(q_probs_1d.clamp_min(0).to(dtype=torch.float32), num_samples=1).squeeze(0)
    delta = (delta / z).to(dtype=torch.float32)
    return torch.multinomial(delta, num_samples=1).squeeze(0)


def sample_with_top_p(logits: torch.Tensor, top_p: float = 0.95, temperature: float = 0.0):
    """
    Mirrors the sampling helper used in the Fast-dLLM v2 remote modeling code.

    Args:
        logits: [B, L, V]
    Returns:
        x_1: [B, L] sampled token ids
        p_1t: [B, L, V] probabilities
    """
    if temperature and temperature > 0:
        scaled_logits = logits / float(temperature)
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > float(top_p)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0)

        probs_sum = torch.sum(probs, dim=-1, keepdim=True).clamp_min(1e-12)
        p_1t = probs / probs_sum

        b, l, v = p_1t.shape
        x_1 = torch.multinomial(p_1t.reshape(b * l, v), num_samples=1).reshape(b, l)
        return x_1, p_1t

    p_1t = torch.softmax(logits, dim=-1)
    x_1 = p_1t.argmax(dim=-1)
    return x_1, p_1t


def _find_mask_spans_1d(mask_1d: torch.Tensor) -> list[list[int]]:
    """
    Find all contiguous spans of True in a 1D boolean tensor.
    Returns a list of spans, each span is a list of indices (L2R).
    """
    assert mask_1d.dim() == 1
    spans: list[list[int]] = []
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


def _estimate_token_acceptance_probs(
    *,
    estimator: str,
    span_logits: torch.Tensor,
    temperature: float,
    top_p: float,
    ssd_confidence_margin_threshold: float,
    ssd_entropy_temperature: float,
) -> torch.Tensor:
    probs = _probs_from_logits(
        span_logits,
        temperature=temperature,
        top_p=top_p,
    )
    if estimator == "hard_margin_threshold":
        if probs.shape[-1] >= 2:
            top2_vals = torch.topk(probs, k=2, dim=-1).values
            confidence_margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            confidence_margin = probs[:, 0]
        return (confidence_margin > float(ssd_confidence_margin_threshold)).to(dtype=probs.dtype)
    if estimator == "soft_entropy_negexp":
        if probs.shape[-1] <= 1:
            normalized_entropy = torch.zeros((probs.shape[0],), device=probs.device, dtype=probs.dtype)
        else:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            normalized_entropy = entropy / torch.log(
                torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype)
            )
        return torch.exp(-float(ssd_entropy_temperature) * normalized_entropy)
    if estimator == "soft_renyi_2_entropy":
        return (probs ** 2).sum(dim=-1).clamp(0.0, 1.0)
    raise ValueError(f"Invalid token acceptance estimator: {estimator}")


def _estimate_expected_accepted_tokens(alpha: torch.Tensor) -> float:
    if alpha.numel() == 0:
        return 0.0
    alpha = alpha.clamp(0.0, 1.0)
    return float(torch.cumprod(alpha, dim=0).sum().item())


def _linear_bin_idx(value: float, *, vmin: float, vmax: float, num_bins: int) -> int:
    if num_bins <= 1 or vmax <= vmin:
        return 0
    value = min(max(float(value), float(vmin)), float(vmax))
    ratio = (value - float(vmin)) / (float(vmax) - float(vmin))
    idx = int(ratio * num_bins)
    return min(num_bins - 1, max(0, idx))


def _mean_normalized_entropy_from_logits(
    span_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> float:
    if span_logits.numel() == 0:
        return 0.0
    probs = _probs_from_logits(
        span_logits,
        temperature=temperature,
        top_p=top_p,
    )
    if probs.shape[-1] <= 1:
        return 0.0
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / torch.log(
        torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype)
    )
    return float(normalized_entropy.mean().item())


def _ucb_context_bucket_idx(
    *,
    span_length: int,
    block_length: int,
    block_progress: float,
    mean_normalized_entropy: float,
    span_length_bins: int,
    block_progress_bins: int,
    entropy_bins: int,
) -> int:
    span_bin = _linear_bin_idx(
        float(span_length),
        vmin=1.0,
        vmax=float(max(1, block_length)),
        num_bins=span_length_bins,
    )
    progress_bin = _linear_bin_idx(
        float(block_progress),
        vmin=0.0,
        vmax=1.0,
        num_bins=block_progress_bins,
    )
    entropy_bin = _linear_bin_idx(
        float(mean_normalized_entropy),
        vmin=0.0,
        vmax=1.0,
        num_bins=entropy_bins,
    )
    return (span_bin * block_progress_bins + progress_bin) * entropy_bins + entropy_bin


def _compute_do_verify_score(
    *,
    score_type: str,
    span_logits: torch.Tensor,
    token_acceptance_estimator: str,
    temperature: float,
    top_p: float,
    ssd_confidence_margin_threshold: float,
    ssd_entropy_temperature: float,
    score_penalty_coef: float,
    sampled_token_confidence: torch.Tensor,
    segment_mask: torch.Tensor,
    confidence_threshold: float,
) -> float:
    sampled_token_confidence = sampled_token_confidence.reshape(-1)
    segment_mask = segment_mask.reshape(-1)
    span_alpha = _estimate_token_acceptance_probs(
        estimator=token_acceptance_estimator,
        span_logits=span_logits,
        temperature=temperature,
        top_p=top_p,
        ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
        ssd_entropy_temperature=ssd_entropy_temperature,
    )
    expected_accepted_tokens = _estimate_expected_accepted_tokens(span_alpha)
    c = float(score_penalty_coef)
    if score_type == "difference_dynamic":
        high_conf_mask = (sampled_token_confidence > float(confidence_threshold)) & segment_mask
        num_high_confidence = int(high_conf_mask.sum().item())
        return expected_accepted_tokens - c * float(num_high_confidence)
    if score_type == "difference_static":
        return expected_accepted_tokens - c
    raise ValueError(f"Invalid do_verify_score_type: {score_type}")


def _decide_do_verify(
    *,
    do_verify_policy: str,
    span_rel: list[int],
    span_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    min_ssd_span_length: int,
    # Score-based policy options.
    do_verify_score_threshold: float,
    hysteresis_threshold_on: float,
    hysteresis_threshold_off: float,
    do_verify_score_type: str,
    score_penalty_coef: float,
    token_acceptance_estimator: str,
    ssd_confidence_margin_threshold: float,
    ssd_entropy_temperature: float,
    sampled_token_confidence: torch.Tensor,
    segment_mask: torch.Tensor,
    confidence_threshold: float,
    # UCB state/options.
    ucb_state: dict | None,
    hysteresis_state: dict | None,
    ucb_beta: float,
    ucb_span_bins: int,
    ucb_progress_bins: int,
    ucb_ent_bins: int,
    ucb_entropy_source: str,
    block_unmasked_count: int,
    block_length: int,
    # Sampling transform context.
    temperature: float,
    top_p: float,
) -> tuple[bool, int | None, int | None]:
    do_verify = True
    chosen_action = None
    chosen_bucket = None
    if do_verify_policy == "mask_span_length":
        do_verify = len(span_rel) >= int(min_ssd_span_length)
    elif do_verify_policy in {"score_threshold", "score_hysteresis"}:
        do_verify_score = float("-inf")
        if span_logits.numel() > 0:
            do_verify_score = _compute_do_verify_score(
                score_type=do_verify_score_type,
                span_logits=span_logits,
                token_acceptance_estimator=token_acceptance_estimator,
                temperature=temperature,
                top_p=top_p,
                ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
                ssd_entropy_temperature=ssd_entropy_temperature,
                score_penalty_coef=score_penalty_coef,
                sampled_token_confidence=sampled_token_confidence,
                segment_mask=segment_mask,
                confidence_threshold=confidence_threshold,
            )
        if do_verify_policy == "score_threshold":
            do_verify = do_verify_score >= float(do_verify_score_threshold)
        else:
            if hysteresis_state is None:
                raise ValueError("hysteresis_state must be initialized for score_hysteresis policy")
            if not (float(hysteresis_threshold_on) > float(hysteresis_threshold_off)):
                raise ValueError("hysteresis_threshold_on must be greater than hysteresis_threshold_off")
            is_on = bool(hysteresis_state["is_on"])
            if is_on:
                do_verify = not (do_verify_score < float(hysteresis_threshold_off))
            else:
                do_verify = bool(do_verify_score > float(hysteresis_threshold_on))
            hysteresis_state["is_on"] = bool(do_verify)
    elif do_verify_policy == "contextual_bandit_ucb":
        if ucb_state is None:
            raise ValueError("ucb_state must be initialized for contextual_bandit_ucb policy")
        ucb_state["decision_t"] = int(ucb_state["decision_t"]) + 1

        if ucb_entropy_source == "span":
            entropy_logits = span_logits
        elif ucb_entropy_source == "masked":
            entropy_logits = masked_logits
        else:
            raise ValueError(f"Invalid ucb_entropy_source: {ucb_entropy_source}")

        mean_entropy = _mean_normalized_entropy_from_logits(
            entropy_logits,
            temperature=temperature,
            top_p=top_p,
        ) if entropy_logits.numel() > 0 else 0.0
        block_progress = float(block_unmasked_count) / float(max(1, block_length))
        block_progress = min(1.0, max(0.0, block_progress))
        chosen_bucket = _ucb_context_bucket_idx(
            span_length=max(1, len(span_rel)),
            block_length=block_length,
            block_progress=block_progress,
            mean_normalized_entropy=mean_entropy,
            span_length_bins=ucb_span_bins,
            block_progress_bins=ucb_progress_bins,
            entropy_bins=ucb_ent_bins,
        )
        log_t = math.log(max(1, int(ucb_state["decision_t"])))
        count = ucb_state["count"]
        reward_sum = ucb_state["reward_sum"]
        scores = []
        for action in (0, 1):
            n_ab = count[action][chosen_bucket]
            if n_ab == 0:
                scores.append(float("inf"))
                continue
            mu_ab = reward_sum[action][chosen_bucket] / float(n_ab)
            bonus = 0.0 if log_t <= 0.0 else float(ucb_beta) * math.sqrt(log_t / float(n_ab))
            scores.append(mu_ab + bonus)
        chosen_action = 1 if scores[1] >= scores[0] else 0
        do_verify = bool(chosen_action == 1)
    else:
        raise ValueError(f"Invalid do_verify_policy: {do_verify_policy}")
    return do_verify, chosen_action, chosen_bucket


def _update_ucb_reward(
    *,
    ucb_state: dict | None,
    chosen_action: int | None,
    chosen_bucket: int | None,
    decoded_count: int,
    do_verify: bool,
) -> None:
    if ucb_state is None or chosen_action is None or chosen_bucket is None:
        return
    time_cost = 2.0 if do_verify else 1.0
    reward = float(decoded_count) / time_cost
    ucb_state["count"][chosen_action][chosen_bucket] += 1
    ucb_state["reward_sum"][chosen_action][chosen_bucket] += reward


@torch.no_grad()
def generate_ssd_policy(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: Optional[int] = None,
    max_gen_toks: Optional[int] = None,
    max_length: Optional[int] = None,
    tokenizer=None,
    mask_id: int = 151665,
    threshold: float = 1.0,
    small_block_size: int = 8,
    block_size: int = 32,
    stop_token: int = 151645,
    stopping_criteria=None,
    top_p: float = 0.95,
    temperature: float = 0.0,
    use_block_cache: bool = False,
    use_ssd_cache: bool = False,
    ssd_ratio_tempering_factor: float = 1.0,
    min_ssd_span_length: int = 1,
    allow_resample: bool = True,
    return_decoding_order: bool = False,
    return_dict_in_generate: bool = False,
    output_scores: bool = False,
    output_hidden_states: bool = False,
    cache_ver: bool = False,
    draft_ver: bool = False,
    use_attention_mask: bool = False,
    do_verify_policy: str = 'mask_span_length',
    do_verify_score_threshold: float = 0.0,
    hysteresis_threshold_on: float = 0.0,
    hysteresis_threshold_off: float = -1.0,
    token_acceptance_estimator: str = "hard_margin_threshold",
    ssd_confidence_margin_threshold: float = 0.05,
    ssd_entropy_temperature: float = 1.0,
    ucb_beta: float = 1.0,
    ucb_span_length_bins: int = 2,
    ucb_block_progress_bins: int = 2,
    ucb_entropy_bins: int = 2,
    ucb_entropy_source: str = "span",
    do_verify_score_type: str = "difference_dynamic",
    score_penalty_coef: float = 2.0,
    **kwargs,
):
    """
    Fast-dLLM v2 generation with Self-Speculative Decoding (SSD).

    Draft: same block-diffusion sub-block sampling as `generate()`.
    Verify: run the same model in "verification mode" with `block_size=1` (causal),
    using the model's right-shifted logits, and apply SSD accept/reject + residual resampling.

    Notes:
    - Verification uses the model's internal causal mask when `block_size=1` (no 2L trick needed).
    - This implementation assumes batch size 1 (consistent with v2 demo scripts).
    """
    # lm_eval passes `max_gen_toks`; treat it as `max_new_tokens` when provided.
    if max_gen_toks is not None:
        max_new_tokens = int(max_gen_toks)
    if max_new_tokens is None and max_length is None and max_gen_toks is None:
        raise ValueError("Either max_new_tokens/max_gen_toks or max_length must be specified")
    if max_new_tokens is None:
        max_new_tokens = int(max_length) - int(input_ids.shape[1])
    
    full_length = math.ceil((input_ids.shape[1] + max_new_tokens) / block_size) * int(block_size)
    full_num_blocks = full_length // int(block_size)

    if int(input_ids.shape[0]) != 1:
        raise ValueError("generate_ssd currently supports batch_size=1")

    device = _get_device(model, input_ids)
    input_ids = input_ids.to(device)

    scores_list = [] if output_scores else None
    decoder_hidden_states = [] if output_hidden_states else None
    decoding_order = [] if return_decoding_order else None
    total_forward_steps = 0

    num_blocks = int(max_new_tokens) // int(block_size)
    original_input_length = int(input_ids.shape[1])

    use_attention_mask = use_attention_mask or draft_ver
    full_attention_mask = None
    curr_attn_mask = None
    if use_attention_mask:
        full_attention_mask = torch.kron(torch.tril(torch.ones(full_num_blocks, full_num_blocks, device=device, dtype=torch.long)), \
            torch.ones(block_size, block_size, device=device, dtype=torch.long)).to(dtype=torch.bool)
        full_verify_attention_mask = torch.tril(torch.ones(full_length, full_length, device=device, dtype=torch.bool))

    # Prefill (same as generate)
    if input_ids.shape[1] > block_size:
        output = model(
            input_ids=input_ids[:, : (input_ids.shape[1] // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=1 if cache_ver else block_size,
            # block_size=block_size,
        )
        total_forward_steps += 1
        logits, past_key_values = output.logits, output.past_key_values
        if output_scores:
            scores_list.append(logits)
        if output_hidden_states and hasattr(output, "hidden_states"):
            decoder_hidden_states.append(output.hidden_states)
        if input_ids.shape[1] % block_size == 0:
            next_token = logits[:, -1:, :].argmax(dim=-1)
            if return_decoding_order:
                abs_pos = int(input_ids.shape[1])
                decoding_order.append([-(float(abs_pos) + 0.5)])
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        past_key_values = None

    num_small_blocks = int(block_size) // int(small_block_size)
    ucb_span_bins = max(1, int(ucb_span_length_bins))
    ucb_progress_bins = max(1, int(ucb_block_progress_bins))
    ucb_ent_bins = max(1, int(ucb_entropy_bins))
    hysteresis_state = {"is_on": False} if do_verify_policy == "score_hysteresis" else None
    ucb_state = None
    if do_verify_policy == "contextual_bandit_ucb":
        ucb_num_buckets = ucb_span_bins * ucb_progress_bins * ucb_ent_bins
        ucb_state = {
            "count": [[0 for _ in range(ucb_num_buckets)] for _ in range(2)],
            "reward_sum": [[0.0 for _ in range(ucb_num_buckets)] for _ in range(2)],
            "decision_t": 0,
        }

    for _block_idx in range(num_blocks):
        if (input_ids[:, original_input_length:] == stop_token).any():
            break

        prompt_length = int(input_ids.shape[1])
        x_init = mask_id * torch.ones(
            (1, block_size - (prompt_length % block_size)),
            device=device,
            dtype=torch.long,
        )
        x_init = torch.cat([input_ids, x_init], dim=1)

        x_t = x_init.clone()
        block_past_key_values = None

        while True:
            if (x_t[:, prompt_length:] == stop_token).any():
                stop_pos = (x_t[:, prompt_length:] == stop_token).nonzero(as_tuple=False)[0][1].item()
                if (x_t[:, prompt_length : prompt_length + stop_pos] == mask_id).sum() == 0:
                    break

            mask_idx = x_t[:, -block_size:] == mask_id

            # Finalize a complete block, update cache, then append one AR token.
            if mask_idx.sum() == 0:
                # if use_attention_mask:
                #     curr_attn_mask = full_verify_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]] if cache_ver \
                #         else full_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]]
                output = model(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=1 if cache_ver else block_size,
                )
                total_forward_steps += 1
                logits, past_key_values = output.logits, output.past_key_values
                if output_scores:
                    scores_list.append(logits)
                if output_hidden_states and hasattr(output, "hidden_states"):
                    decoder_hidden_states.append(output.hidden_states)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                if return_decoding_order:
                    abs_pos = int(x_t.shape[1])
                    decoding_order.append([-(float(abs_pos) + 0.5)])
                x_t = torch.cat([x_t, next_token], dim=1)
                break

            # Sub-block draft + verify (left-to-right)
            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size

                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                while True:
                    mask_idx = x_t[:, -block_size:] == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    if (x_t[:, prompt_length:] == stop_token).any():
                        stop_pos = (x_t[:, prompt_length:] == stop_token).nonzero(as_tuple=False)[0][1].item()
                        if (x_t[:, prompt_length : prompt_length + stop_pos] == mask_id).sum() == 0:
                            break

                    # Draft forward: same as `generate()`
                    seg_mask_sub = mask_idx[:, start:end][0]  # (seg_len,) force batch size 1
                    # In SSD, if we already have cached K/V for earlier accepted tokens within this small block,
                    # we can start the forward from the first still-masked position and patch the block cache
                    # starting at that replace_position. This is gated by `use_ssd_cache`.
                    first_mask_rel_in_seg = 0
                    mask_pos = seg_mask_sub.nonzero(as_tuple=True)[0]
                    if mask_pos.numel() > 0:
                        first_mask_rel_in_seg = int(mask_pos.min().item())

                    # Effective segment start (skip prefix that is already accepted / unmasked)
                    start_eff = start + (first_mask_rel_in_seg if (use_block_cache and use_ssd_cache) else 0)
                    seg_mask = seg_mask_sub[(first_mask_rel_in_seg if (use_block_cache and use_ssd_cache) else 0) :]
                    seg_base = int(block_size + start_eff)  # map segment index -> [0, block_size)

                    if use_block_cache:
                        # Initialize block cache with a full-block forward when needed.
                        if block_past_key_values is None:
                            if draft_ver:
                                curr_attn_mask = full_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]]
                                if first_mask_rel_in_seg > 0:
                                    curr_ver_attn_mask = full_verify_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]]
                                    curr_attn_mask[:first_mask_rel_in_seg, :] = curr_ver_attn_mask[:first_mask_rel_in_seg, :]
                            output = model(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_size=block_size if not draft_ver else None,
                                attention_mask=curr_attn_mask if draft_ver else None,
                            )
                            total_forward_steps += 1
                            logits_draft, block_past_key_values = output.logits, output.block_past_key_values
                            logits_draft = _shift_logits_right(logits_draft)
                            logits_draft = logits_draft[:, start_eff:end]
                        else:
                            # Normal behavior: forward the whole small block span.
                            # SSD-cache behavior (use_ssd_cache=True): forward only from the first masked token onward
                            # and patch `block_past_key_values` at `replace_position=first_mask_idx`.
                            if use_ssd_cache:
                                raise NotImplementedError("use_ssd_cache is not implemented")
                                replace_pos = int(small_block_start_idx + first_mask_rel_in_seg)
                                output = model(
                                    input_ids=x_t[:, (-block_size + replace_pos) : end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=replace_pos,
                                )
                                total_forward_steps += 1
                                logits_draft = _shift_logits_right(output.logits)
                            else:
                                if draft_ver:
                                    sub_start = x_t.shape[1]+start
                                    sub_end = x_t.shape[1]+end if end is not None else x_t.shape[1]
                                    curr_attn_mask = full_attention_mask[sub_start:sub_end, :sub_end]
                                    if first_mask_rel_in_seg > 0:
                                        curr_ver_attn_mask = full_verify_attention_mask[sub_start:sub_end, :sub_end]
                                        curr_attn_mask[:first_mask_rel_in_seg, :] = curr_ver_attn_mask[:first_mask_rel_in_seg, :]
                                output = model(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                )
                                total_forward_steps += 1
                                logits_draft = _shift_logits_right(output.logits)
                    else:
                        if draft_ver:
                            curr_attn_mask = full_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]]
                            if first_mask_rel_in_seg > 0:
                                curr_ver_attn_mask = full_verify_attention_mask[x_t.shape[1]-block_size:x_t.shape[1], :x_t.shape[1]]
                                curr_attn_mask[:first_mask_rel_in_seg, :] = curr_ver_attn_mask[:first_mask_rel_in_seg, :]
                        output = model(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                            # block_size=block_size,
                            block_size=block_size if not draft_ver else None,
                            attention_mask=curr_attn_mask if draft_ver else None,
                        )
                        total_forward_steps += 1
                        logits_draft = _shift_logits_right(output.logits)
                        logits_draft = logits_draft[:, start_eff:end]

                    if output_scores:
                        scores_list.append(logits_draft)
                    if output_hidden_states and hasattr(output, "hidden_states"):
                        decoder_hidden_states.append(output.hidden_states)

                    if hasattr(model, "sample_with_top_p"):
                        x_1, p_1t = model.sample_with_top_p(logits_draft, top_p=top_p, temperature=temperature)
                    else:
                        x_1, p_1t = sample_with_top_p(logits_draft, top_p=top_p, temperature=temperature)
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )

                    # Build first contiguous L2R mask span inside this (effective) segment.
                    mask_pos = seg_mask.nonzero(as_tuple=True)[0]
                    span_rel: list[int] = []
                    if mask_pos.numel() > 0:
                        i0 = int(mask_pos.min().item())
                        j = i0
                        while j < int(seg_mask.shape[0]) and bool(seg_mask[j].item()):
                            span_rel.append(int(j))
                            j += 1

                    span_rel_t = torch.tensor(span_rel, device=device, dtype=torch.long)
                    if span_rel_t.numel() > 0:
                        span_logits = logits_draft[0].index_select(0, span_rel_t)
                    else:
                        span_logits = torch.empty((0, logits_draft.shape[-1]), device=logits_draft.device, dtype=logits_draft.dtype)
                    if do_verify_policy == "contextual_bandit_ucb" and ucb_entropy_source == "masked":
                        # UCB "masked" entropy context should reflect the whole current sub-block window.
                        masked_rel_t = seg_mask_sub.nonzero(as_tuple=True)[0]
                        if masked_rel_t.numel() > 0:
                            masked_logits = logits_draft[0].index_select(0, masked_rel_t)
                        else:
                            masked_logits = torch.empty((0, logits_draft.shape[-1]), device=logits_draft.device, dtype=logits_draft.dtype)
                    else:
                        masked_logits = torch.empty((0, logits_draft.shape[-1]), device=logits_draft.device, dtype=logits_draft.dtype)
                    # UCB progress context should reflect the current sub-block window.
                    segment_unmasked_count = int((~seg_mask_sub).sum().item())

                    do_verify, chosen_action, chosen_bucket = _decide_do_verify(
                        do_verify_policy=do_verify_policy,
                        span_rel=span_rel,
                        span_logits=span_logits,
                        masked_logits=masked_logits,
                        min_ssd_span_length=int(min_ssd_span_length),
                        do_verify_score_threshold=do_verify_score_threshold,
                        hysteresis_threshold_on=hysteresis_threshold_on,
                        hysteresis_threshold_off=hysteresis_threshold_off,
                        do_verify_score_type=do_verify_score_type,
                        score_penalty_coef=score_penalty_coef,
                        token_acceptance_estimator=token_acceptance_estimator,
                        ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
                        ssd_entropy_temperature=ssd_entropy_temperature,
                        sampled_token_confidence=x1_p[0],
                        segment_mask=seg_mask,
                        confidence_threshold=threshold,
                        ucb_state=ucb_state,
                        hysteresis_state=hysteresis_state,
                        ucb_beta=ucb_beta,
                        ucb_span_bins=ucb_span_bins,
                        ucb_progress_bins=ucb_progress_bins,
                        ucb_ent_bins=ucb_ent_bins,
                        ucb_entropy_source=ucb_entropy_source,
                        block_unmasked_count=segment_unmasked_count,
                        block_length=int(seg_mask_sub.shape[0]),
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    # Fallback: if span too small, keep the original confidence-thresholding behavior.
                    if not do_verify:
                        x1_p = torch.where(seg_mask.unsqueeze(0), x1_p, -torch.inf)
                        unmask_idx = x1_p > float(threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=device), max_prob_idx] = True
                        x_t[:, start_eff:end][unmask_idx] = x_1[unmask_idx]
                        _update_ucb_reward(
                            ucb_state=ucb_state,
                            chosen_action=chosen_action,
                            chosen_bucket=chosen_bucket,
                            decoded_count=int(unmask_idx.sum().item()),
                            do_verify=False,
                        )
                        if return_decoding_order:
                            step_decoding_positions = []
                            block_start_abs = int(x_t.shape[1] - block_size)
                            rel_pos = unmask_idx[0].nonzero(as_tuple=True)[0].tolist()
                            for r in rel_pos:
                                abs_pos = block_start_abs + seg_base + int(r)
                                step_decoding_positions.append(-int(abs_pos))
                            if step_decoding_positions:
                                decoding_order.append(step_decoding_positions)
                        continue
                    
                    # NOTE: do verification step
                    draft_tokens = x_1[0, span_rel_t].to(dtype=torch.long)  # (S,)
                    p_probs = p_1t[0, span_rel_t, :].to(dtype=torch.float32)  # (S, V)
                    p_sel = p_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)  # (S,)

                    # Verification forward (block_size=1 causal mask). We do NOT use block_cache here because
                    # the remote code disables attention_mask when block_cache is populated.
                    span_pos_in_block = (span_rel_t + seg_base).to(dtype=torch.long)  # (S,)
                    # We only need the block prefix up to `end` for causal verification:
                    # tokens after `end` cannot affect logits within this sub-block span.
                    ver_seq = x_t[:, -block_size:end].clone()
                    # Fill the entire sub-block's masked positions with the draft tokens for proper causal context.
                    ver_seq[:, seg_base : seg_base + int(seg_mask.shape[0])][seg_mask.unsqueeze(0)] = x_1[0][seg_mask].unsqueeze(0)

                    out_ver = model(
                        input_ids=ver_seq,
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=False,
                        block_size=1,
                    )
                    total_forward_steps += 1
                    logits_ver = _shift_logits_right(out_ver.logits)[0, span_pos_in_block, :]  # (S, V)
                    q_probs = _probs_from_logits(logits_ver, temperature=temperature, top_p=top_p)  # (S, V)
                    q_sel = q_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)  # (S,)

                    if float(ssd_ratio_tempering_factor) != 1.0:
                        ratios = (q_sel / p_sel) ** float(ssd_ratio_tempering_factor)
                    else:
                        ratios = q_sel / p_sel
                    ratios = ratios.clamp(max=1.0)

                    rand = torch.rand((ratios.shape[0],), device=device, dtype=ratios.dtype)
                    accept_flags = rand < ratios
                    reject_idx = (~accept_flags).nonzero(as_tuple=True)[0]
                    first_reject = int(reject_idx[0].item()) if reject_idx.numel() > 0 else None

                    # Apply updates: accepted prefix, and optionally residual-resample at first rejection.
                    update_rel: list[int] = []
                    resampled_rel = None
                    decoded_for_reward = 0
                    if first_reject is None:
                        update_rel = span_rel
                    else:
                        if first_reject > 0:
                            update_rel.extend(span_rel[:first_reject])
                        if allow_resample:
                            resampled_rel = int(span_rel[first_reject])
                            update_rel.append(resampled_rel)

                    if update_rel:
                        upd_rel_t = torch.tensor(update_rel, device=device, dtype=torch.long)
                        # Default: use draft tokens
                        x_t[:, start_eff:end][0, upd_rel_t] = x_1[0, upd_rel_t]

                        # If rejected and resampling enabled: override the rejected position.
                        if first_reject is not None and allow_resample:
                            rej_rel = int(span_rel[first_reject])
                            rej_q = q_probs[first_reject]  # (V,)
                            rej_p = p_probs[first_reject]  # (V,)
                            resampled = _reject_resample_from_delta(rej_q, rej_p)
                            x_t[:, start_eff:end][0, rej_rel] = resampled.to(dtype=torch.long)
                        decoded_for_reward = int(len(update_rel))
                        if return_decoding_order:
                            step_decoding_positions = []
                            block_start_abs = int(x_t.shape[1] - block_size)
                            for rel in update_rel:
                                abs_pos = block_start_abs + seg_base + int(rel)
                                if resampled_rel is not None and int(rel) == int(resampled_rel):
                                    step_decoding_positions.append(float(abs_pos) + 0.5)
                                else:
                                    step_decoding_positions.append(int(abs_pos))
                            if step_decoding_positions:
                                decoding_order.append(step_decoding_positions)
                    else:
                        # Avoid stalling when `allow_resample=False` and the first token is rejected.
                        # Fall back to the original diffusion-style confidence thresholding, which also
                        # forces at least one token (the max-prob position) to be unmasked.
                        x1_p = torch.squeeze(
                            torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                        )
                        x1_p = torch.where(seg_mask.unsqueeze(0), x1_p, -torch.inf)
                        unmask_idx = x1_p > float(threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=device), max_prob_idx] = True
                        x_t[:, start_eff:end][unmask_idx] = x_1[unmask_idx]
                        decoded_for_reward = int(unmask_idx.sum().item())
                        if return_decoding_order:
                            step_decoding_positions = []
                            block_start_abs = int(x_t.shape[1] - block_size)
                            rel_pos = unmask_idx[0].nonzero(as_tuple=True)[0].tolist()
                            for r in rel_pos:
                                abs_pos = block_start_abs + seg_base + int(r)
                                step_decoding_positions.append(-int(abs_pos))
                            if step_decoding_positions:
                                decoding_order.append(step_decoding_positions)
                    _update_ucb_reward(
                        ucb_state=ucb_state,
                        chosen_action=chosen_action,
                        chosen_bucket=chosen_bucket,
                        decoded_count=decoded_for_reward,
                        do_verify=True,
                    )

        input_ids = x_t

    if (input_ids[:, original_input_length:] == stop_token).any():
        stop_pos = (input_ids[:, original_input_length:] == stop_token).nonzero(as_tuple=False)[0][1].item()
        input_ids = input_ids[:, : original_input_length + stop_pos + 1]

    out = (
        GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=tuple(scores_list) if output_scores and scores_list else None,
            hidden_states=tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
        )
        if return_dict_in_generate
        else input_ids
    )
    if return_decoding_order:
        stats = {
            "decoding_order": decoding_order,
            "total_forward_steps": int(total_forward_steps),
        }
        return out, stats
    return out

