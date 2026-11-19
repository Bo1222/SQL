"""GRPO / DAPO training math and rollout contracts.

This module is deliberately independent of any trainer, model or distributed
backend.  It is the single place where the following are defined and tested:

* group-relative advantages (GRPO / Dr. GRPO);
* DAPO dynamic-sampling group selection (filter all-same groups, bounded
  resampling plan);
* token-level PPO policy loss with asymmetric clip-higher;
* KL estimators used by GRPO;
* DAPO overlong-reward shaping;
* multi-turn action/observation token alignment.

The RL trainer must not re-implement these formulas.  In particular, the old
``train/grpo*.py`` code is archived and is *not* a valid reference: it used a
causal log-probability shifted by one position, included generation padding in
the loss, and was single-turn / LoRA-only.

Reference definitions:
- GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

import torch


class RLError(ValueError):
    """Raised when a rollout or loss input violates the RL contract."""


# ---------------------------------------------------------------------------
# Group advantages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupStats:
    groups_total: int
    groups_mixed: int
    groups_all_zero: int
    groups_all_correct: int
    groups_constant_other: int
    rewards_mean: float
    rewards_std: float
    mixed_rate: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _group_values(rewards: Sequence[float], group_ids: Sequence[Hashable]):
    if len(rewards) != len(group_ids):
        raise RLError("rewards and group_ids must have the same length")
    if not rewards:
        raise RLError("empty group batch")
    groups: dict[Hashable, list[float]] = {}
    order: list[Hashable] = []
    for reward, group_id in zip(rewards, group_ids):
        value = float(reward)
        if not torch.isfinite(torch.tensor(value)):
            raise RLError(f"non-finite reward in group {group_id!r}: {reward!r}")
        if group_id not in groups:
            groups[group_id] = []
            order.append(group_id)
        groups[group_id].append(value)
    return groups, order


def group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
    allow_singleton: bool = False,
    unbiased: bool = True,
) -> tuple[list[float], GroupStats]:
    """Return GRPO/Dr. GRPO advantages aligned with ``rewards``.

    The standard GRPO advantage is ``(r - mean_g) / (std_g + eps)``.
    ``unbiased=True`` uses the sample standard deviation (PyTorch /
    veRL default, denominator ``n-1``); ``unbiased=False`` uses the
    population standard deviation (the old project trainer, denominator
    ``n``).  For n=4 the two differ by ``sqrt(4/3)`` in scale, so do not mix
    them when comparing runs.  ``normalize_by_std=False`` gives the Dr. GRPO /
    DAPO-style mean-centered advantage, which avoids inflating low-variance
    groups.

    A group whose rewards are all equal has zero advantage.  Such groups carry
    no relative signal and should normally be removed by dynamic sampling
    before the update; returning zeros here is a safety net, not a substitute
    for filtering.
    """
    groups, order = _group_values(rewards, group_ids)
    advantages: dict[Hashable, list[float]] = {}
    mixed = all_zero = all_correct = constant_other = 0
    for group_id in order:
        values = groups[group_id]
        if len(values) < 2 and not allow_singleton:
            raise RLError(
                f"group {group_id!r} has {len(values)} rollout(s); GRPO needs at least 2"
            )
        tensor = torch.tensor(values, dtype=torch.float64)
        mean = float(tensor.mean())
        std = float(tensor.std(unbiased=unbiased))
        if std <= eps:
            advantages[group_id] = [0.0] * len(values)
            if all(value == 0.0 for value in values):
                all_zero += 1
            elif all(value == 1.0 for value in values):
                all_correct += 1
            else:
                constant_other += 1
        else:
            mixed += 1
            if normalize_by_std:
                advantages[group_id] = [float((value - mean) / (std + eps)) for value in values]
            else:
                advantages[group_id] = [float(value - mean) for value in values]

    flat = [value for group_id in order for value in groups[group_id]]
    flat_tensor = torch.tensor(flat, dtype=torch.float64)
    stats = GroupStats(
        groups_total=len(order),
        groups_mixed=mixed,
        groups_all_zero=all_zero,
        groups_all_correct=all_correct,
        groups_constant_other=constant_other,
        rewards_mean=float(flat_tensor.mean()),
        rewards_std=float(flat_tensor.std(unbiased=False)),
        mixed_rate=mixed / max(len(order), 1),
    )
    aligned = [advantages[group_id].pop(0) for group_id in group_ids]
    return aligned, stats


# ---------------------------------------------------------------------------
# DAPO dynamic sampling
# ---------------------------------------------------------------------------


@dataclass
class RolloutGroup:
    """One prompt group; ``payload`` is opaque trainer data (tokens, masks...)."""

    group_id: Hashable
    rewards: Sequence[float]
    payload: Any = None


@dataclass(frozen=True)
class DynamicSamplingStats:
    groups_sampled: int
    groups_kept: int
    groups_filtered: int
    groups_not_used: int
    all_zero: int
    all_correct: int
    constant_other: int
    target_groups: int
    underfilled: bool
    filter_rate: float
    mixed_rate: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DynamicSamplingResult:
    kept: list[RolloutGroup] = field(default_factory=list)
    stats: DynamicSamplingStats | None = None


def select_mixed_groups(
    groups: Sequence[RolloutGroup],
    *,
    target_groups: int,
    max_groups: int | None = None,
) -> DynamicSamplingResult:
    """DAPO dynamic-sampling selection over already-generated groups.

    This function only decides which groups enter the update and reports the
    resampling budget.  It does **not** generate new rollouts.  A trainer that
    wants true dynamic sampling must call it inside its rollout loop and keep
    sampling new prompts until ``target_groups`` mixed groups are found or the
    ``max_groups`` budget is exhausted.

    ``groups`` is the generated prefix in sampling order.  Groups not needed
    after the target is reached are counted as ``not_used`` rather than
    filtered; they were generated but would waste an update slot.
    """
    if target_groups < 1:
        raise RLError("target_groups must be positive")
    if not groups:
        raise RLError("no groups generated")
    sizes = {len(group.rewards) for group in groups}
    if len(sizes) != 1:
        raise RLError(f"all groups must have the same rollout count, got {sorted(sizes)}")
    if min(sizes) < 2:
        raise RLError("dynamic sampling requires n >= 2 rollouts per group")
    budget = len(groups) if max_groups is None else int(max_groups)
    if budget < target_groups:
        raise RLError("max_groups must be >= target_groups")
    budget = min(budget, len(groups))

    kept: list[RolloutGroup] = []
    all_zero = all_correct = constant_other = filtered = 0
    sampled = 0
    for group in groups[:budget]:
        sampled += 1
        values = [float(value) for value in group.rewards]
        low, high = min(values), max(values)
        if high > low:
            if len(kept) < target_groups:
                kept.append(group)
            # A mixed group after the target is reached is not used.
            continue
        if all(value == 0.0 for value in values):
            all_zero += 1
        elif all(value == 1.0 for value in values):
            all_correct += 1
        else:
            constant_other += 1
        filtered += 1

    not_used = sampled - len(kept) - filtered
    underfilled = len(kept) < target_groups
    stats = DynamicSamplingStats(
        groups_sampled=sampled,
        groups_kept=len(kept),
        groups_filtered=filtered,
        groups_not_used=max(not_used, 0),
        all_zero=all_zero,
        all_correct=all_correct,
        constant_other=constant_other,
        target_groups=target_groups,
        underfilled=underfilled,
        filter_rate=filtered / max(sampled, 1),
        mixed_rate=len(kept) / max(sampled, 1),
    )
    return DynamicSamplingResult(kept=kept, stats=stats)


# ---------------------------------------------------------------------------
# Token-level policy loss
# ---------------------------------------------------------------------------


def _check_tensor(name: str, value: torch.Tensor, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RLError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise RLError(f"{name} must be {ndim}-D, got shape {tuple(value.shape)}")
    return value


def _broadcast_advantages(advantages: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if not isinstance(advantages, torch.Tensor):
        raise RLError("advantages must be a torch.Tensor")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    if advantages.ndim != 2:
        raise RLError(f"advantages must be [B] or [B,T], got {tuple(advantages.shape)}")
    if advantages.shape[0] != shape[0]:
        raise RLError("advantages batch size does not match log-probabilities")
    if advantages.shape[1] == 1:
        return advantages.expand(shape)
    if advantages.shape != shape:
        raise RLError(
            f"advantages shape {tuple(advantages.shape)} does not match {tuple(shape)}"
        )
    return advantages


def policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    agg_mode: str = "token-mean",
    global_num_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO/GRPO/DAPO policy loss on action tokens only.

    ``log_probs`` and ``old_log_probs`` are current and behavior-policy
    log-probabilities per response token.  ``response_mask`` is 1 only for
    model-generated action tokens (including invalid actions) and 0 for tool
    observations / padding.  DAPO clip-higher is expressed by
    ``clip_low=0.2, clip_high=0.28``.

    ``agg_mode``:
      * ``token-mean``: every valid action token has equal weight (DAPO);
      * ``seq-mean-token-mean``: per-trajectory token mean, then equal
        trajectory weight (the GRPO convention used by the old trainer);
      * ``seq-mean-token-sum``: per-trajectory token sum, then equal
        trajectory weight.

    ``global_num_tokens`` is used by distributed callers so the local loss is
    the global token mean; the caller is responsible for the usual FSDP
    gradient scaling.  This reference implementation does not include veRL's
    optional dual-clip lower bound (``clip_ratio_c``); the trainer uses veRL's
    full policy-loss implementation.
    """
    log_probs = _check_tensor("log_probs", log_probs, 2)
    old_log_probs = _check_tensor("old_log_probs", old_log_probs, 2)
    response_mask = _check_tensor("response_mask", response_mask, 2)
    if log_probs.shape != old_log_probs.shape or log_probs.shape != response_mask.shape:
        raise RLError(
            "log_probs/old_log_probs/response_mask shape mismatch: "
            f"{tuple(log_probs.shape)} vs {tuple(old_log_probs.shape)} vs "
            f"{tuple(response_mask.shape)}"
        )
    if clip_low < 0 or clip_high < 0 or clip_low >= 1 or not torch.isfinite(torch.tensor(clip_high)):
        raise RLError("invalid clip range")
    if agg_mode not in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum"):
        raise RLError(f"unknown agg_mode: {agg_mode}")

    mask = response_mask.to(torch.float32)
    if not torch.all((mask == 0) | (mask == 1)):
        raise RLError("response_mask must contain only 0/1")
    if float(mask.sum()) <= 0:
        raise RLError("response_mask has no action tokens")
    adv = _broadcast_advantages(advantages, log_probs.shape).to(log_probs.dtype)

    negative_approx_kl = (log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -adv * ratio
    pg_losses2 = -adv * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    loss_mat = torch.maximum(pg_losses1, pg_losses2)

    valid = mask.sum()
    if agg_mode == "token-mean":
        denominator = float(global_num_tokens) if global_num_tokens else float(valid)
        loss = (loss_mat * mask).sum() / max(denominator, 1.0)
    elif agg_mode == "seq-mean-token-mean":
        seq_tokens = mask.sum(dim=-1)
        seq_losses = (loss_mat * mask).sum(dim=-1) / seq_tokens.clamp_min(1.0)
        seq_present = (seq_tokens > 0).to(torch.float32)
        loss = (seq_losses * seq_present).sum() / seq_present.sum().clamp_min(1.0)
    else:
        seq_tokens = mask.sum(dim=-1)
        seq_losses = (loss_mat * mask).sum(dim=-1)
        seq_present = (seq_tokens > 0).to(torch.float32)
        loss = (seq_losses * seq_present).sum() / seq_present.sum().clamp_min(1.0)

    with torch.no_grad():
        clipfrac = ((pg_losses2 > pg_losses1).to(torch.float32) * mask).sum() / valid
        approx_kl = (-(log_probs - old_log_probs).detach() * mask).sum() / valid
        ratio_mean = (ratio.detach() * mask).sum() / valid
    metrics = {
        "loss": float(loss.detach()),
        "valid_tokens": float(valid),
        "clipfrac": float(clipfrac),
        "approx_kl": float(approx_kl),
        "ratio_mean": float(ratio_mean),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# KL estimators and DAPO overlong shaping
# ---------------------------------------------------------------------------


def kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    kind: str = "low_var_kl",
) -> torch.Tensor:
    """Masked policy-vs-reference KL estimator.

    ``kind="low_var_kl"`` is the non-negative k3 estimator
    ``exp(ref-logp) - (ref-logp) - 1`` used by veRL's GRPO.  ``kind="kl"`` is
    the (possibly negative) k1 estimator.  The returned scalar is the masked
    mean over action tokens.
    """
    log_probs = _check_tensor("log_probs", log_probs, 2)
    ref_log_probs = _check_tensor("ref_log_probs", ref_log_probs, 2)
    response_mask = _check_tensor("response_mask", response_mask, 2)
    if log_probs.shape != ref_log_probs.shape or log_probs.shape != response_mask.shape:
        raise RLError("KL tensors have mismatched shapes")
    # Match veRL: k1/k2 use policy-minus-reference; k3 uses
    # reference-minus-policy and is clamped for numerical stability.
    delta = log_probs - ref_log_probs
    if kind == "kl":
        values = delta
    elif kind == "abs":
        values = delta.abs()
    elif kind == "mse":
        values = 0.5 * delta.pow(2)
    elif kind == "low_var_kl":
        log_ratio = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        values = (torch.exp(log_ratio) - log_ratio - 1.0).clamp(-10.0, 10.0)
    else:
        raise RLError(f"unknown KL kind: {kind}")
    mask = response_mask.to(torch.float32)
    if float(mask.sum()) <= 0:
        raise RLError("response_mask has no action tokens")
    return (values * mask).sum() / mask.sum()


def overlong_penalty(
    response_length: int,
    *,
    max_response_length: int,
    buffer_length: int,
    penalty_factor: float = 1.0,
) -> float:
    """DAPO overlong reward shaping, matching ``DAPORewardManager``.

    Responses shorter than ``max_response_length - buffer_length`` are not
    penalised.  After that, the penalty grows linearly.  Within the valid
    response range (``response_length <= max_response_length``) the penalty
    reaches ``-penalty_factor`` at the maximum length.  This matches
    ``verl.workers.reward_manager.dapo.DAPORewardManager``; it intentionally
    does not add an extra lower clamp for out-of-range inputs.
    """
    if max_response_length <= 0 or buffer_length <= 0 or buffer_length > max_response_length:
        raise RLError("invalid overlong buffer configuration")
    if response_length < 0:
        raise RLError("response_length must be non-negative")
    expected = max_response_length - buffer_length
    exceed = response_length - expected
    if exceed <= 0:
        return 0.0
    return min(-(exceed / buffer_length) * penalty_factor, 0.0)


# ---------------------------------------------------------------------------
# Multi-turn token alignment
# ---------------------------------------------------------------------------


def segments_to_response(segments: Sequence[dict]) -> tuple[list[int], list[int]]:
    """Build response ids/mask from runtime segments.

    Every ``action`` segment is a policy decision and gets mask 1, including
    invalid actions: RL must be able to learn from them.  Every ``observation``
    segment is environment feedback and gets mask 0.  This intentionally
    differs from the SFT mask, which supervises only successful actions.
    """
    ids: list[int] = []
    mask: list[int] = []
    for index, segment in enumerate(segments):
        kind = segment.get("kind")
        segment_ids = list(segment.get("ids") or [])
        if kind == "prompt":
            # ``runtime.EpisodeState.record`` prepends the prompt segment; it is
            # already represented by ``prompt_ids`` and must not enter the
            # response mask.
            continue
        if not segment_ids:
            continue
        if kind == "action":
            ids.extend(segment_ids)
            mask.extend([1] * len(segment_ids))
        elif kind == "observation":
            ids.extend(segment_ids)
            mask.extend([0] * len(segment_ids))
        else:
            raise RLError(f"unexpected response segment kind at {index}: {kind!r}")
    return ids, mask


def trajectory_tensors(record: dict) -> dict[str, Any]:
    """Normalise a runtime rollout record into training tensors.

    The record must contain ``prompt_ids`` and ``segments``.  The returned
    ``response_mask`` marks action tokens, not observation/prompt tokens.
    """
    prompt_ids = list(record.get("prompt_ids") or [])
    segments = record.get("segments")
    if not prompt_ids:
        raise RLError("record has no prompt_ids")
    if not isinstance(segments, list) or not segments:
        raise RLError("record has no segments")
    response_ids, response_mask = segments_to_response(segments)
    if not response_ids:
        raise RLError("record has no response tokens")
    if not any(response_mask):
        raise RLError("record has no action tokens")
    input_ids = prompt_ids + response_ids
    result = {
        "input_ids": input_ids,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
    }
    validate_training_tensors(
        input_ids=input_ids,
        response_mask=response_mask,
        prompt_length=len(prompt_ids),
    )
    return result


def action_spans(segments: Sequence[dict]) -> list[tuple[int, int]]:
    """Return half-open action-token spans in response coordinates."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in segments:
        kind = segment.get("kind")
        length = len(segment.get("ids") or [])
        if kind == "prompt":
            continue
        if kind == "action" and length:
            spans.append((offset, offset + length))
        offset += length
    return spans


def validate_training_tensors(
    *,
    input_ids: Sequence[int],
    response_mask: Sequence[int],
    prompt_length: int,
    old_log_probs: Sequence[float] | None = None,
    log_probs: Sequence[float] | None = None,
    response_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Check the token/mask/log-prob alignment required by the policy loss."""
    if prompt_length < 0 or prompt_length > len(input_ids):
        raise RLError("prompt_length is outside input_ids")
    response_length = len(input_ids) - prompt_length
    if response_length <= 0:
        raise RLError("zero-length response")
    if len(response_mask) != response_length:
        raise RLError(
            f"response_mask length {len(response_mask)} != response length {response_length}"
        )
    if any(value not in (0, 1) for value in response_mask):
        raise RLError("response_mask must contain only 0/1")
    if not any(response_mask):
        raise RLError("response_mask has no action tokens")
    if response_ids is not None and len(response_ids) != response_length:
        raise RLError("response_ids length does not match response length")
    for name, values in (("old_log_probs", old_log_probs), ("log_probs", log_probs)):
        if values is None:
            continue
        if len(values) != response_length:
            raise RLError(f"{name} length {len(values)} != response length {response_length}")
        tensor = torch.tensor(list(values), dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise RLError(f"{name} contains non-finite values")
    return {
        "prompt_length": prompt_length,
        "response_length": response_length,
        "action_tokens": int(sum(response_mask)),
        "observation_tokens": int(response_length - sum(response_mask)),
    }
