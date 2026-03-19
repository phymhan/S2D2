import argparse
import math
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict
import random
import time
# global_step = 0

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


def _probs_from_logits(logits_2d: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    Convert logits -> proposal/target probs using the same sampling transforms as draft sampling.

    logits_2d: (N, V)
    returns: (N, V) float probs
    """
    if temperature < 1e-6:  # NOTE: treat as 0, temperature=0 is treated as =1 with do_sample=False
        return F.softmax(logits_2d, dim=-1, dtype=torch.float32)
    if temperature != 1.0:
        logits_2d = logits_2d / temperature
    if top_k and top_k > 0:
        logits_2d = top_k_logits(logits_2d, top_k)
    if top_p is not None and top_p < 1.0:
        logits_2d = top_p_logits(logits_2d, top_p)
    return F.softmax(logits_2d, dim=-1, dtype=torch.float32)


def _reject_resample_from_delta(q_probs_1d: torch.Tensor, p_probs_1d: torch.Tensor) -> torch.Tensor:
    """
    Residual resampling used in SSD: sample from normalized (q - p)+.
    Falls back to sampling from q if the residual mass is ~0.
    """
    delta = (q_probs_1d - p_probs_1d).clamp_min(0.0)
    z = float(delta.sum().item())
    if z <= 0.0 or not torch.isfinite(delta).all():
        # Fallback: sample from q directly.
        return torch.multinomial(q_probs_1d.clamp_min(0).to(dtype=torch.float32), num_samples=1).squeeze(0)
    delta = (delta / z).to(dtype=torch.float32)
    return torch.multinomial(delta, num_samples=1).squeeze(0)


def _construct_2l_verifier_attention_mask_bool(*, L: int, cache_len: int, device: torch.device) -> torch.Tensor:
    """
    2L trick attention mask for SDAR inference path (bool mask where True means "allowed").

    Keys are laid out as: [cached keys (cache_len) | first-half (L) | second-half masks (L)]
    Queries are: [first-half (L) | second-half masks (L)]
    """
    # (2L, cache_len + 2L)
    attn = torch.zeros((2 * L, cache_len + 2 * L), device=device, dtype=torch.bool)
    if cache_len > 0:
        attn[:, :cache_len] = True

    # First half: token-causal within first-half.
    attn[:L, cache_len:cache_len + L] = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=0)

    # Second half: strict-causal into first-half (exclude same-position draft token).
    attn[L:, cache_len:cache_len + L] = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=-1)

    # Second half: block-diagonal self-attention (block_size=1 => identity).
    attn[L:, cache_len + L:cache_len + 2 * L] = torch.eye(L, device=device, dtype=torch.bool)
    return attn.unsqueeze(0)


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
    sampled_token_confidence: torch.Tensor,
    span_logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    ssd_confidence_margin_threshold: float,
    ssd_confidence_power: float,
    ssd_entropy_threshold: float,
    ssd_confidence_margin_coef: float,
    ssd_entropy_temperature: float,
) -> torch.Tensor:
    need_probs = estimator in {
        "hard_margin_threshold",
        "hard_entropy_threshold",
        "soft_clipped_linear_margin",
        "soft_entropy_negexp",
        "soft_renyi_2_entropy",
    }
    probs = None
    if need_probs:
        probs = _probs_from_logits(
            span_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # (S, V)

    if estimator == "hard_margin_threshold":
        if probs.shape[-1] >= 2:
            top2_vals = torch.topk(probs, k=2, dim=-1).values
            confidence_margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            confidence_margin = probs[:, 0]
        return (confidence_margin > ssd_confidence_margin_threshold).to(dtype=sampled_token_confidence.dtype)
    if estimator == "hard_entropy_threshold":
        if probs.shape[-1] <= 1:
            normalized_entropy = torch.zeros((probs.shape[0],), device=probs.device, dtype=probs.dtype)
        else:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            normalized_entropy = entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype))
        return (normalized_entropy < ssd_entropy_threshold).to(dtype=sampled_token_confidence.dtype)
    if estimator == "soft_confidence_power":
        return sampled_token_confidence.clamp(0.0, 1.0) ** ssd_confidence_power
    if estimator == "soft_clipped_linear_margin":
        if probs.shape[-1] >= 2:
            top2_vals = torch.topk(probs, k=2, dim=-1).values
            confidence_margin = top2_vals[:, 0] - top2_vals[:, 1]
        else:
            confidence_margin = probs[:, 0]
        return (ssd_confidence_margin_coef * confidence_margin).clamp(0.0, 1.0)
    if estimator == "soft_entropy_negexp":
        if probs.shape[-1] <= 1:
            normalized_entropy = torch.zeros((probs.shape[0],), device=probs.device, dtype=probs.dtype)
        else:
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            normalized_entropy = entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype))
        return torch.exp(-ssd_entropy_temperature * normalized_entropy)
    if estimator == "soft_renyi_2_entropy":
        return (probs ** 2).sum(dim=-1).clamp(0.0, 1.0)
    raise ValueError(f"Invalid token acceptance estimator: {estimator}")


def _estimate_expected_accepted_tokens(alpha: torch.Tensor) -> float:
    """
    Estimate E[K] = sum_{k=1}^{L} prod_{i=1}^{k} alpha_i for a token span.
    """
    if alpha.numel() == 0:
        return 0.0
    alpha = alpha.clamp(0.0, 1.0)
    return float(torch.cumprod(alpha, dim=0).sum().item())


def _linear_bin_idx(value: float, *, vmin: float, vmax: float, num_bins: int) -> int:
    if num_bins <= 1:
        return 0
    if vmax <= vmin:
        return 0
    value = min(max(float(value), float(vmin)), float(vmax))
    ratio = (value - float(vmin)) / (float(vmax) - float(vmin))
    idx = int(ratio * num_bins)
    return min(num_bins - 1, max(0, idx))


def _mean_normalized_entropy_from_logits(
    span_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> float:
    if span_logits.numel() == 0:
        return 0.0
    probs = _probs_from_logits(
        span_logits,
        temperature=temperature,
        top_k=top_k,
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
    sampled_token_confidence: torch.Tensor,
    mask_index: torch.Tensor,
    x0_p: torch.Tensor,
    confidence_threshold: float,
    token_acceptance_estimator: str,
    temperature: float,
    top_k: int,
    top_p: float,
    ssd_confidence_margin_threshold: float,
    ssd_confidence_power: float,
    ssd_entropy_threshold: float,
    ssd_confidence_margin_coef: float,
    ssd_entropy_temperature: float,
    score_penalty_coef: float,
) -> float:
    if score_type in {"difference_dynamic", "difference_static"}:
        span_alpha = _estimate_token_acceptance_probs(
            estimator=token_acceptance_estimator,
            sampled_token_confidence=sampled_token_confidence,
            span_logits=span_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
            ssd_confidence_power=ssd_confidence_power,
            ssd_entropy_threshold=ssd_entropy_threshold,
            ssd_confidence_margin_coef=ssd_confidence_margin_coef,
            ssd_entropy_temperature=ssd_entropy_temperature,
        )
        expected_accepted_tokens = _estimate_expected_accepted_tokens(span_alpha)
        c = float(score_penalty_coef)
        if score_type == "difference_dynamic":
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            high_conf_mask = confidence[0] > confidence_threshold
            num_high_confidence = int(high_conf_mask.sum().item())
            return expected_accepted_tokens - c * num_high_confidence
        if score_type == "difference_static":
            # global global_step
            # global_step += 1
            # print(f"global_step: {global_step}, expected_accepted_tokens: {expected_accepted_tokens}")
            return expected_accepted_tokens - c
    raise ValueError(f"Invalid do_verify_score_type: {score_type}")


def _decide_do_verify(
    *,
    # Policy selection + basic context.
    do_verify_policy: str,
    span_rel: list[int],
    span_rel_t: torch.Tensor,
    span_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    mask_index: torch.Tensor,
    x0_p: torch.Tensor,
    block_unmasked_count: int,
    block_length: int,
    min_ssd_span_length: int,
    legacy_ssd_span_strategy: bool,
    confidence_threshold: float,

    # Score-based policy options.
    do_verify_score_threshold: float,
    hysteresis_threshold_on: float,
    hysteresis_threshold_off: float,
    do_verify_score_type: str,
    score_penalty_coef: float,
    token_acceptance_estimator: str,
    ssd_confidence_margin_threshold: float,
    ssd_confidence_power: float,
    ssd_entropy_threshold: float,
    ssd_confidence_margin_coef: float,
    ssd_entropy_temperature: float,

    # UCB state/options.
    ucb_state: Dict[str, object] | None,
    hysteresis_state: Dict[str, object] | None,
    ucb_beta: float,
    ucb_span_bins: int,
    ucb_progress_bins: int,
    ucb_ent_bins: int,
    ucb_entropy_source: str,

    # Sampling transform context (used by some policies / estimators).
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[bool, int | None, int | None]:
    do_verify = True
    chosen_action = None
    chosen_bucket = None
    if do_verify_policy == 'mask_span_length':
        if (len(span_rel) < min_ssd_span_length) and (not legacy_ssd_span_strategy or \
            (torch.where(mask_index, x0_p, -torch.inf) > confidence_threshold).sum() >= \
                min(min_ssd_span_length, mask_index.sum().item())):
            do_verify = False
    elif do_verify_policy in {'score_threshold', 'score_hysteresis'}:
        do_verify_score = float("-inf")
        if span_logits.numel() > 0:
            do_verify_score = _compute_do_verify_score(
                score_type=do_verify_score_type,
                span_logits=span_logits,
                sampled_token_confidence=x0_p[0, span_rel_t].clamp(0.0, 1.0),
                mask_index=mask_index,
                x0_p=x0_p,
                confidence_threshold=confidence_threshold,
                token_acceptance_estimator=token_acceptance_estimator,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
                ssd_confidence_power=ssd_confidence_power,
                ssd_entropy_threshold=ssd_entropy_threshold,
                ssd_confidence_margin_coef=ssd_confidence_margin_coef,
                ssd_entropy_temperature=ssd_entropy_temperature,
                score_penalty_coef=score_penalty_coef,
            )
            # print(f"do_verify_score: {do_verify_score}")
        if do_verify_policy == "score_threshold":
            if do_verify_score < do_verify_score_threshold:
                do_verify = False
            # print(f"global_step: {global_step}, do_verify: {do_verify}, do_verify_score: {do_verify_score}")
        else:
            if hysteresis_state is None:
                raise ValueError("hysteresis_state must be initialized for score_hysteresis policy")
            if not (hysteresis_threshold_on > hysteresis_threshold_off):
                raise ValueError("hysteresis_threshold_on must be greater than hysteresis_threshold_off")
            is_on = bool(hysteresis_state["is_on"])
            if is_on:
                do_verify = not (do_verify_score < hysteresis_threshold_off)
            else:
                do_verify = bool(do_verify_score >= hysteresis_threshold_on)
            hysteresis_state["is_on"] = bool(do_verify)
    elif do_verify_policy == 'contextual_bandit_ucb':
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
            top_k=top_k,
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
        raise ValueError(f"Invalid verify policy: {do_verify_policy}")
    return do_verify, chosen_action, chosen_bucket


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
        cache_ver=False,
        draft_ver=False,
        min_ssd_span_length=1,
        # ssd_ratio_relax_factor=1.0,
        ssd_ratio_tempering_factor=1.0,
        ssd_ratio_threshold=-1.0,
        ssd_threshold_all_spans=False,
        always_check_high_confidence=False,
        legacy_ssd_span_strategy=False,
        allow_resample=True,

        # SSD policy options
        do_verify_policy='mask_span_length',
        do_verify_score_threshold=-1.0,
        hysteresis_threshold_on=0.0,
        hysteresis_threshold_off=-1.0,
        token_acceptance_estimator="hard_margin_threshold",
        ssd_confidence_margin_threshold=0.05,
        ssd_confidence_power=1.0,
        ssd_entropy_threshold=0.5,
        ssd_confidence_margin_coef=1.0,
        ssd_entropy_temperature=1.0,
        ucb_beta=1.0,
        ucb_span_length_bins=2,
        ucb_block_progress_bins=2,
        ucb_entropy_bins=2,
        ucb_entropy_source="span",
        do_verify_score_type="difference_dynamic",
        score_penalty_coef=2.0,
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
    verifier_attention_mask = torch.tril(torch.ones(total_length, total_length, device=model.device)).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    total_forward_steps = 0

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_ver_attn_mask = verifier_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_ver_attn_mask if cache_ver else cur_attn_mask,
            # attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True)
        total_forward_steps += 1

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)
    block_unmasked_count = [0 if i < prefill_blocks else -1 for i in range(num_blocks)]  # NOTE: -1: not visited, >= 0: unmasked count
    ucb_span_bins = max(1, int(ucb_span_length_bins))
    ucb_progress_bins = max(1, int(ucb_block_progress_bins))
    ucb_ent_bins = max(1, int(ucb_entropy_bins))
    hysteresis_state = None
    if do_verify_policy == "score_hysteresis":
        hysteresis_state = {"is_on": False}
    ucb_state = None
    masked_logits = None
    if do_verify_policy == "contextual_bandit_ucb":
        ucb_num_buckets = ucb_span_bins * ucb_progress_bins * ucb_ent_bins
        ucb_state = {
            "count": [[0 for _ in range(ucb_num_buckets)] for _ in range(2)],  # action: 0=no-verify, 1=verify
            "reward_sum": [[0.0 for _ in range(ucb_num_buckets)] for _ in range(2)],
            "decision_t": 0,
        }

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_ver_attn_mask = verifier_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block * block_length:(num_block+1)*block_length]
        block_unmasked_count[num_block] = 0
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(
                    cur_x,
                    attention_mask=cur_ver_attn_mask if cache_ver else cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True)
                total_forward_steps += 1
                break

            # Denosing
            if draft_ver:
                first_mask_index_rel = _find_mask_spans_1d(mask_index[0])[0][0]
                if first_mask_index_rel > 0:
                    cur_attn_mask[:, :first_mask_index_rel, :] = cur_ver_attn_mask[:, :first_mask_index_rel, :]
            logits = model(
                cur_x,
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

            # # NOTE: debugging
            # if tokenizer is not None:
            #     print(f"block: {num_block}, step: {step}")
            #     print(f"cur_x: {cur_x[0].tolist()} {tokenizer.decode(cur_x[0].tolist()).replace('<|MASK|>', '[M]')}")
            #     print(f"x0: {x0[0].tolist()} {tokenizer.decode(x0[0].tolist())}")
            #     print(f"x0_p: {[round(float(v), 3) for v in x0_p[0].tolist()]}")
            #     print(f"\n")

            # Self-speculative decoding (SSD) with block_size=1 verifier (static: one block at a time).
            # - draft: current block diffusion sampling (x0, x0_p)
            # - verify: 2L trick + rejection sampling over first contiguous mask span
            # - fallback: if SSD decodes < num_transfer_tokens[step], use original remasking_strategy
            min_k = min(int(num_transfer_tokens[step]), mask_index.sum().item())
            decoded_this_step = 0
            transfer_index = torch.zeros_like(cur_x, dtype=torch.bool)
            update_tokens = cur_x.clone()
            step_decoding_positions: List[float] = ['|'] if block_unmasked_count[num_block] == 0 else []
            
            # Draft-fill the entire current block for verifier's first half.
            input_seq_ver_first = cur_x.clone()
            if mask_index.any():
                input_seq_ver_first[mask_index] = x0[mask_index]

            # Find first contiguous mask span within this block (L2R), capped at target_k.
            spans_rel = _find_mask_spans_1d(mask_index[0])
            span_rel: list[int] = spans_rel[0] if len(spans_rel) > 0 else []
            span_rel_t = torch.tensor(span_rel, device=cur_x.device, dtype=torch.long)

            if span_rel_t.numel() > 0:
                span_logits = logits[0].index_select(0, span_rel_t)  # (S, V)
            else:
                span_logits = torch.empty((0, logits.shape[-1]), device=logits.device, dtype=logits.dtype)
            if do_verify_policy == "contextual_bandit_ucb" and ucb_entropy_source == "masked":
                masked_rel_t = mask_index[0].nonzero(as_tuple=True)[0]
                if masked_rel_t.numel() > 0:
                    masked_logits = logits[0].index_select(0, masked_rel_t)  # (M, V)
                else:
                    masked_logits = torch.empty((0, logits.shape[-1]), device=logits.device, dtype=logits.dtype)
            # else:
            #     masked_logits = torch.empty((0, logits.shape[-1]), device=logits.device, dtype=logits.dtype)

            do_verify, chosen_action, chosen_bucket = _decide_do_verify(
                # Policy selection + core context.
                do_verify_policy=do_verify_policy,
                span_rel=span_rel,
                span_rel_t=span_rel_t,
                span_logits=span_logits,
                masked_logits=masked_logits,
                mask_index=mask_index,
                x0_p=x0_p,
                block_unmasked_count=block_unmasked_count[num_block],
                block_length=block_length,
                min_ssd_span_length=min_ssd_span_length,
                legacy_ssd_span_strategy=legacy_ssd_span_strategy,
                confidence_threshold=confidence_threshold,

                # Score-based policy options.
                do_verify_score_threshold=do_verify_score_threshold,
                hysteresis_threshold_on=hysteresis_threshold_on,
                hysteresis_threshold_off=hysteresis_threshold_off,
                do_verify_score_type=do_verify_score_type,
                score_penalty_coef=score_penalty_coef,
                token_acceptance_estimator=token_acceptance_estimator,
                ssd_confidence_margin_threshold=ssd_confidence_margin_threshold,
                ssd_confidence_power=ssd_confidence_power,
                ssd_entropy_threshold=ssd_entropy_threshold,
                ssd_confidence_margin_coef=ssd_confidence_margin_coef,
                ssd_entropy_temperature=ssd_entropy_temperature,

                # UCB policy options/state.
                ucb_state=ucb_state,
                hysteresis_state=hysteresis_state,
                ucb_beta=ucb_beta,
                ucb_span_bins=ucb_span_bins,
                ucb_progress_bins=ucb_progress_bins,
                ucb_ent_bins=ucb_ent_bins,
                ucb_entropy_source=ucb_entropy_source,

                # Sampling transform context.
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            resampled_rel: list[int] = []
            if do_verify:
                L = int(cur_x.shape[1])
                cache_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0

                # 2L verifier input: [draft-filled | all-mask], take logits from masked half.
                mask_tokens = torch.full_like(input_seq_ver_first, mask_id)
                input_seq_ver = torch.cat([input_seq_ver_first, mask_tokens], dim=1)  # (1, 2L)

                verify_attn_mask = _construct_2l_verifier_attention_mask_bool(
                    L=L, cache_len=cache_len, device=cur_x.device
                )  # (1, 2L, cache_len+2L)
                verify_position_ids = torch.cat([cur_position_ids, cur_position_ids], dim=1)  # (1, 2L)

                verify_logits = model(
                    input_seq_ver,
                    attention_mask=verify_attn_mask,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits
                total_forward_steps += 1

                q_logits = verify_logits[0, span_rel_t + L, :]  # (S, V) from masked half
                q_probs = _probs_from_logits(q_logits, temperature=temperature, top_k=top_k, top_p=top_p)  # (S, V)

                draft_tokens = x0[0, span_rel_t]  # (S,)
                p_sel = x0_p[0, span_rel_t].clamp_min(1e-12)  # (S,)
                q_sel = q_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)  # (S,)
                # ratios = (q_sel / p_sel * ssd_ratio_relax_factor).clamp(max=1.0)
                if ssd_ratio_tempering_factor != 1.0:
                    ratios = (q_sel / p_sel) ** ssd_ratio_tempering_factor
                else:
                    ratios = q_sel / p_sel
                ratios = (ratios).clamp(max=1.0)

                accept_flags = torch.rand((ratios.shape[0],), device=ratios.device, dtype=ratios.dtype) < ratios
                reject_idx = (~accept_flags).nonzero(as_tuple=True)[0]
                first_reject = int(reject_idx[0].item()) if reject_idx.numel() > 0 else None
                # p_all = x0_p[0]
                # q_probs_all = _probs_from_logits(verify_logits[0, L:, :], temperature=temperature, top_k=top_k, top_p=top_p)
                # q_all = q_probs_all.gather(-1, input_seq_ver_first[0].unsqueeze(-1)).squeeze(-1)
                # print(round(expected_accepted_tokens), first_reject+1 if first_reject is not None else len(span_rel))

                # Build an ordered update list (L2R): accepted prefix, and optionally the first rejected position.
                update_rel: list[int] = []
                if first_reject is None:
                    # All accepted.
                    update_rel = span_rel
                else:
                    # Accept prefix.
                    if first_reject > 0:
                        update_rel.extend(span_rel[:first_reject])

                    # At first rejection, residual-resample and always unmask it.
                    rej_rel = int(span_rel[first_reject])
                    update_rel.append(rej_rel)

                if len(update_rel) > 0:
                    upd_rel_t = torch.tensor(update_rel, device=cur_x.device, dtype=torch.long)
                    transfer_index[0, upd_rel_t] = True
                    update_tokens[0, upd_rel_t] = x0[0, upd_rel_t]

                    # If there was a rejection, override the rejected token with residual-resample.
                    if allow_resample and (first_reject is not None and rej_rel in update_rel):
                        rej_q = q_probs[first_reject]  # (V,)
                        rej_p = _probs_from_logits(
                            logits[0, rej_rel, :].unsqueeze(0),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )[0]
                        resampled = _reject_resample_from_delta(rej_q, rej_p)
                        update_tokens[0, rej_rel] = resampled.to(dtype=torch.long)
                        resampled_rel.append(rej_rel)

                    decoded_this_step += int(len(update_rel))

                    # Track decoding order for SSD updates (accepted vs resampled).
                    if return_forward_stats:
                        base_abs = int(num_block * block_length)
                        for r in update_rel:
                            abs_pos = base_abs + int(r)
                            if int(r) in set(resampled_rel):
                                step_decoding_positions.append(float(abs_pos) + 0.5)
                            else:
                                step_decoding_positions.append(int(abs_pos))
                
                # Optional: threshold accept additional tokens by ratio.
                # - If `ssd_threshold_all_spans=False`, threshold only within the first span.
                # - If `ssd_threshold_all_spans=True`, threshold across all remaining masked positions in this block.
                thresholded_rel: list[int] = []
                if ssd_ratio_threshold is not None and float(ssd_ratio_threshold) >= 0.0:
                    thr = float(ssd_ratio_threshold)
                    if not ssd_threshold_all_spans:
                        # `ratios` is aligned with `span_rel`.
                        for i, rel in enumerate(span_rel):
                            if bool(transfer_index[0, rel].item()):
                                continue
                            if float(ratios[i].item()) >= thr:
                                transfer_index[0, rel] = True
                                update_tokens[0, rel] = x0[0, rel]
                                decoded_this_step += 1
                                thresholded_rel.append(int(rel))
                    else:
                        cand_rel = (mask_index[0] & (~transfer_index[0])).nonzero(as_tuple=True)[0].tolist()
                        if cand_rel:
                            rel_t = torch.tensor(cand_rel, device=cur_x.device, dtype=torch.long)
                            q_logits_c = verify_logits[0, rel_t + L, :]  # (C, V) from masked half
                            q_probs_c = _probs_from_logits(
                                q_logits_c, temperature=temperature, top_k=top_k, top_p=top_p
                            )  # (C, V)
                            draft_tok_c = x0[0, rel_t]  # (C,)
                            p_sel_c = x0_p[0, rel_t].clamp_min(1e-12)  # (C,)
                            q_sel_c = q_probs_c.gather(-1, draft_tok_c.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)  # (C,)
                            if ssd_ratio_tempering_factor != 1.0:
                                ratios_c = (q_sel_c / p_sel_c) ** ssd_ratio_tempering_factor
                            else:
                                ratios_c = q_sel_c / p_sel_c
                            ratios_c = (ratios_c).clamp(max=1.0)  # (C,)
                            for rel, rr in zip(cand_rel, ratios_c.tolist()):
                                if bool(transfer_index[0, rel].item()):
                                    continue
                                if float(rr) >= thr:
                                    transfer_index[0, rel] = True
                                    update_tokens[0, rel] = x0[0, rel]
                                    decoded_this_step += 1
                                    thresholded_rel.append(int(rel))

                    if return_forward_stats and thresholded_rel:
                        base_abs = int(num_block * block_length)
                        for r in thresholded_rel:
                            step_decoding_positions.append(float(base_abs + int(r)) + 0.2)
            
            # NOTE: always checking high confidence tokens for remaining masked positions.
            if always_check_high_confidence:
                remaining_high_confidence_positions: List[List[int]] = [[] for _ in range(cur_x.shape[0])]
                remaining_high_confidence_index = torch.zeros_like(x0, dtype=torch.bool)
                remaining_mask = mask_index & (~transfer_index)
                confidence = torch.where(remaining_mask, x0_p, -torch.inf)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = int(high_conf_mask.sum().item())
                    if num_high_confidence > 0:
                        remaining_high_confidence_index[j] = high_conf_mask
                        sel_pos = high_conf_mask.nonzero(as_tuple=True)[0]
                        if sel_pos.numel() > 0:
                            sel_conf = confidence[j, sel_pos]
                            _, order_idx = torch.sort(sel_conf, descending=True)
                            remaining_high_confidence_positions[j] = [int(sel_pos[t].item()) for t in order_idx.tolist()]

                transfer_index |= remaining_high_confidence_index
                update_tokens[remaining_high_confidence_index] = x0[remaining_high_confidence_index]
                decoded_this_step += int(remaining_high_confidence_index.sum().item())
                if return_forward_stats and bool(remaining_high_confidence_index.any().item()):
                    base_abs = int(num_block * block_length)
                    for r in remaining_high_confidence_positions[0]:
                        step_decoding_positions.append(-(base_abs + int(r))-0.4)
            
            # If SSD didn't decode enough, use existing remasking strategy for remaining masked positions.
            if decoded_this_step < min_k:
                k_left = int(min_k - decoded_this_step)
                remaining_mask = mask_index & (~transfer_index)
                fallback_positions: List[List[int]] = [[] for _ in range(cur_x.shape[0])]

                # Sampling strategy (fallback)
                if remasking_strategy == 'sequential':
                    fallback_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if not remaining_mask[j].any():
                            continue
                        first_mask_index = int(remaining_mask[j].nonzero(as_tuple=True)[0].min().item())
                        picked = 0
                        idx = first_mask_index
                        ordered: List[int] = []
                        while idx < cur_x.shape[1] and picked < k_left:
                            if bool(remaining_mask[j, idx].item()):
                                fallback_index[j, idx] = True
                                ordered.append(int(idx))
                                picked += 1
                            idx += 1
                        fallback_positions[j] = ordered

                elif remasking_strategy == 'low_confidence_static':
                    confidence = torch.where(remaining_mask, x0_p, -torch.inf)
                    fallback_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = min(int(k_left), int(remaining_mask[j].sum().item()))
                        if k <= 0:
                            continue
                        _, idx = torch.topk(confidence[j], k)
                        fallback_index[j, idx] = True
                        fallback_positions[j] = [int(p) for p in idx.tolist()]

                elif remasking_strategy == 'low_confidence_dynamic':
                    confidence = torch.where(remaining_mask, x0_p, -torch.inf)
                    fallback_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = min(int(k_left), int(remaining_mask[j].sum().item()))
                        if k <= 0:
                            continue
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = int(high_conf_mask.sum().item())
                        if num_high_confidence >= k:
                            # Accept ALL tokens above threshold (matches `generate.py` behavior).
                            fallback_index[j] = high_conf_mask
                            # Preserve an order: descending confidence among selected positions.
                            sel_pos = high_conf_mask.nonzero(as_tuple=True)[0]
                            if sel_pos.numel() > 0:
                                sel_conf = confidence[j, sel_pos]
                                _, order_idx = torch.sort(sel_conf, descending=True)
                                fallback_positions[j] = [int(sel_pos[t].item()) for t in order_idx.tolist()]
                        else:
                            _, idx = torch.topk(confidence[j], k)
                            fallback_index[j, idx] = True
                            fallback_positions[j] = [int(p) for p in idx.tolist()]

                elif remasking_strategy == "entropy_bounded":
                    eps = 1e-12
                    entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                    entropies = torch.where(remaining_mask, entropies, torch.inf)
                    fallback_index = torch.zeros_like(x0, dtype=torch.bool)
                    ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=1)
                    for j in range(x0_p.shape[0]):
                        total_masked = int(remaining_mask[j].sum().item())
                        if total_masked <= 0:
                            continue
                        k = int(torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item())
                        k = max(1, min(k, total_masked, int(k_left)))
                        selected_token_indices = order[j, :k]
                        fallback_index[j, selected_token_indices] = True
                        fallback_positions[j] = [int(p) for p in selected_token_indices.tolist()]

                else:
                    raise ValueError(
                        f"Unknown remasking strategy: {remasking_strategy}")

                # Add fallback picks into the same transfer/update tensors.
                transfer_index |= fallback_index
                update_tokens[fallback_index] = x0[fallback_index]
                decoded_this_step += int(fallback_index.sum().item())

                if return_forward_stats and bool(fallback_index.any().item()):
                    base_abs = int(num_block * block_length)
                    for r in fallback_positions[0]:
                        step_decoding_positions.append(-int(base_abs + int(r)))

            if ucb_state is not None and chosen_action is not None and chosen_bucket is not None:
                time_cost = 2.0 if do_verify else 1.0  # verifier uses one extra model forward
                reward = float(decoded_this_step) / time_cost
                ucb_state["count"][chosen_action][chosen_bucket] += 1
                ucb_state["reward_sum"][chosen_action][chosen_bucket] += reward

            # Apply updates once.
            cur_x[transfer_index] = update_tokens[transfer_index]
            block_unmasked_count[num_block] += decoded_this_step

            if return_forward_stats:
                decoding_order.append(step_decoding_positions)

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
    parser.add_argument("--gen_length", type=int, default=2048,
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
    parser.add_argument("--return_forward_stats", "--forward_stats", action='store_true',
                        help="Return forward statistics")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cache_ver", action='store_true')
    parser.add_argument("--draft_ver", action='store_true')
    parser.add_argument("--min_ssd_span_length", type=int, default=1,
                        help="Minimum length of span to invoke SSD")
    # parser.add_argument("--ssd_ratio_relax_factor", type=float, default=1.0,
    #                     help="Relaxation factor for SSD ratio")
    parser.add_argument("--ssd_ratio_tempering_factor", type=float, default=1.0,
                        help="Tempering factor for SSD ratio")
    parser.add_argument("--ssd_ratio_threshold", type=float, default=-1.0,
                        help="If >= 0, threshold-accept tokens with ratio >= this value")
    parser.add_argument("--ssd_threshold_all_spans", action='store_true',
                        help="If set, threshold across all remaining masked positions (not just first span)")
    parser.add_argument("--always_check_high_confidence", action='store_true',
                        help="If set, always check high confidence tokens for remaining masked positions")
    parser.add_argument("--do_verify_policy", type=str, default="mask_span_length",
                        choices=["mask_span_length", "score_threshold", "score_hysteresis", "contextual_bandit_ucb"],
                        help="Policy for verifying whether to invoke SSD")
    parser.add_argument("--do_verify_score_threshold", type=float, default=0.0,
                        help="Threshold for score_threshold policy: do_verify_score < do_verify_score_threshold")
    parser.add_argument("--hysteresis_threshold_on", type=float, default=0.0,
                        help="Hysteresis turn-on threshold: if s > th_on then do verify")
    parser.add_argument("--hysteresis_threshold_off", type=float, default=-1.0,
                        help="Hysteresis turn-off threshold: if s < th_off then stop verify")
    parser.add_argument("--do_verify_score_type", type=str, default="difference_dynamic",
                        choices=["difference_dynamic", "difference_static"],
                        help="Scoring function used by score-based verify policies")
    parser.add_argument("--score_penalty_coef", type=float, default=2.0,
                        help="Penalty coefficient c used by do_verify_score_type")
    parser.add_argument("--ucb_beta", type=float, default=1.0,
                        help="Exploration coefficient beta for contextual_bandit_ucb policy")
    parser.add_argument("--ucb_span_length_bins", type=int, default=2,
                        help="Number of span-length bins in contextual_bandit_ucb policy")
    parser.add_argument("--ucb_block_progress_bins", type=int, default=2,
                        help="Number of block-progress bins in contextual_bandit_ucb policy")
    parser.add_argument("--ucb_entropy_bins", type=int, default=2,
                        help="Number of entropy bins in contextual_bandit_ucb policy")
    parser.add_argument("--ucb_entropy_source", type=str, default="span",
                        choices=["span", "masked"],
                        help="Entropy source for contextual_bandit_ucb policy: first span or all masked tokens")
    parser.add_argument("--token_acceptance_estimator", type=str, default="hard_margin_threshold",
                        choices=[
                            "hard_margin_threshold",
                            "hard_entropy_threshold",
                            "soft_confidence_power",
                            "soft_clipped_linear_margin",
                            "soft_entropy_negexp",
                            "soft_renyi_2_entropy",
                        ],
                        help="Estimator for per-token acceptance probability alpha_i used by SSD expected-acceptance estimation")
    parser.add_argument("--ssd_confidence_margin_threshold", type=float, default=0.05,
                        help="Threshold for hard_margin_threshold estimator: alpha_i = 1[m_i > threshold]")
    parser.add_argument("--ssd_confidence_power", type=float, default=0.8,
                        help="Power for soft_confidence_power estimator: alpha_i = confidence_i ** power")
    parser.add_argument("--ssd_entropy_threshold", type=float, default=0.1,
                        help="Threshold for hard_entropy_threshold estimator: alpha_i = 1[H_i < threshold], H_i is normalized entropy")
    parser.add_argument("--ssd_confidence_margin_coef", type=float, default=1.0,
                        help="Coefficient for soft_clipped_linear_margin estimator: alpha_i = clip(coef * margin_i, 0, 1)")
    parser.add_argument("--ssd_entropy_temperature", type=float, default=1.0,
                        help="Temperature for soft_entropy_negexp estimator: alpha_i = exp(-temp * H_i)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to generate")
    parser.add_argument("--no_chat", action="store_true",
                        help="If set, skip chat template and use raw prompt text")
    parser.add_argument("--legacy_ssd_span_strategy", action='store_true',
                        help="If set, use legacy SSD span strategy")
    parser.add_argument("--no_resample", action='store_true',
                        help="If set, do not resample rejected tokens")
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
    if args.hysteresis_threshold_on <= args.hysteresis_threshold_off:
        parser.error("--hysteresis_threshold_on must be greater than --hysteresis_threshold_off")

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

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
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
        cache_ver=args.cache_ver,
        draft_ver=args.draft_ver,
        min_ssd_span_length=args.min_ssd_span_length,
        # ssd_ratio_relax_factor=args.ssd_ratio_relax_factor,
        ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
        ssd_ratio_threshold=args.ssd_ratio_threshold,
        ssd_threshold_all_spans=args.ssd_threshold_all_spans,
        always_check_high_confidence=args.always_check_high_confidence,
        legacy_ssd_span_strategy=args.legacy_ssd_span_strategy,
        allow_resample=not args.no_resample,
        do_verify_policy=args.do_verify_policy,
        do_verify_score_threshold=args.do_verify_score_threshold,
        hysteresis_threshold_on=args.hysteresis_threshold_on,
        hysteresis_threshold_off=args.hysteresis_threshold_off,
        do_verify_score_type=args.do_verify_score_type,
        score_penalty_coef=args.score_penalty_coef,
        ucb_beta=args.ucb_beta,
        ucb_span_length_bins=args.ucb_span_length_bins,
        ucb_block_progress_bins=args.ucb_block_progress_bins,
        ucb_entropy_bins=args.ucb_entropy_bins,
        ucb_entropy_source=args.ucb_entropy_source,
        token_acceptance_estimator=args.token_acceptance_estimator,
        ssd_confidence_margin_threshold=args.ssd_confidence_margin_threshold,
        ssd_confidence_power=args.ssd_confidence_power,
        ssd_entropy_threshold=args.ssd_entropy_threshold,
        ssd_confidence_margin_coef=args.ssd_confidence_margin_coef,
        ssd_entropy_temperature=args.ssd_entropy_temperature,
    )
    end_time = time.perf_counter()

    if args.return_forward_stats:
        output_ids, stats = output_ids
        decoding_order = stats.get('decoding_order', None)
        total_forward_steps = stats.get('total_forward_steps', 0)
        prompt_length = tokens['input_ids'].shape[1]
        gen_ids = output_ids[:, prompt_length:]
        trimmed_positions = []
        print(f"Decoding steps: {len(decoding_order)}, Total forward steps: {total_forward_steps}")
        if decoding_order is not None:
            print(f"[order] decoding_order has {len(decoding_order)} steps")
            is_new_block = False
            # Print each step as a set of absolute positions updated at that step.
            for i, pos_list in enumerate(decoding_order):
                if len(pos_list) == 0:
                    print(f"-")
                    continue
                if pos_list[0] == '|':
                    is_new_block = True
                    pos_list = pos_list[1:]
                    print(f"[order] step {i}: <NB> {pos_list}")
                else:
                    print(f"[order] step {i}: {pos_list}")  # NOTE: do not sort, should keep the original order
            # NOTE: Debugging, print per-step decoded tokens
            # `decoding_order` contains absolute positions in the output sequence, with markers:
            # - positive int: normal speculative decode at position `pos`, `pos` + .5 resampled, `pos` + .2 thresholded
            # - negative int: forced diffusion decode at position `-pos`
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