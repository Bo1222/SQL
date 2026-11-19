"""Project-local DAPO dynamic-sampling adapter.

The installed veRL commit defines ``algorithm.filter_groups`` but has no
execution path.  This module keeps the filtering logic pure and testable; the
trainer is responsible for generating new batches until the target number of
mixed groups is reached.

Filtering metric is terminal EX (``token_level_scores.sum(-1)``), never a
process reward.  A group with identical terminal rewards has no GRPO signal
and is filtered.  A singleton group is treated as incomplete and rejected; it
is not silently kept.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Hashable, Sequence

from e1_alignment.rl_core import (
    DynamicSamplingResult,
    DynamicSamplingStats,
    RLError,
)


@dataclass(frozen=True)
class GroupSelection:
    indices: list[int]
    stats: DynamicSamplingStats
    uids: list[Hashable]
    unavailable_groups: int = 0
    mixed_groups: int = 0


def group_indices_by_uid(uids: Sequence[Hashable]) -> list[list[int]]:
    """Return trajectory indices grouped by uid, preserving first-seen order."""
    order: list[Hashable] = []
    grouped: dict[Hashable, list[int]] = {}
    for index, uid in enumerate(uids):
        key = uid.item() if hasattr(uid, "item") else uid
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(index)
    return [grouped[key] for key in order]


def select_group_indices(
    uids: Sequence[Hashable],
    rewards: Sequence[float],
    *,
    target_groups: int,
    max_groups: int,
    reward_available: Sequence[bool] | None = None,
    require_mixed: bool = True,
) -> GroupSelection:
    """Select target mixed terminal groups and return flattened indices.

    ``max_groups`` is the total generation budget in groups.  The caller should
    pass ``target_groups * max_num_gen_batches``.  The function does not
    generate; it only evaluates the prefix it is given.
    """
    if len(uids) != len(rewards):
        raise RLError("uids and rewards must have the same length")
    if reward_available is not None and len(uids) != len(reward_available):
        raise RLError("reward availability must align with trajectories")
    if target_groups < 1 or max_groups < target_groups:
        raise RLError("invalid target/max group budget")
    grouped = group_indices_by_uid(uids)
    if not grouped:
        raise RLError("no groups to select")
    sizes = {len(indices) for indices in grouped}
    if len(sizes) != 1:
        raise RLError(f"groups have inconsistent rollout counts: {sorted(sizes)}")
    n = next(iter(sizes))
    if n < 2:
        raise RLError("dynamic sampling requires n >= 2 rollouts per group")
    kept, unavailable = [], 0
    all_zero = all_correct = constant_other = filtered = mixed_sampled = 0
    sampled = min(max_groups, len(grouped))
    for group_index, indices in enumerate(grouped[:sampled]):
        if reward_available is not None and not all(bool(reward_available[i]) for i in indices):
            unavailable += 1
            filtered += 1
            continue
        values = [float(rewards[i]) for i in indices]
        if not all(math.isfinite(value) for value in values):
            raise RLError("non-finite group reward")
        mixed = max(values) > min(values)
        mixed_sampled += int(mixed)
        if not mixed:
            if all(value == 0 for value in values): all_zero += 1
            elif all(value == 1 for value in values): all_correct += 1
            else: constant_other += 1
        if require_mixed and not mixed:
            filtered += 1
            continue
        if len(kept) < target_groups:
            kept.append(group_index)
    flat = [i for index in kept for i in grouped[index]]
    stats = DynamicSamplingStats(groups_sampled=sampled, groups_kept=len(kept),
        groups_filtered=filtered, groups_not_used=sampled-len(kept)-filtered,
        all_zero=all_zero, all_correct=all_correct, constant_other=constant_other,
        target_groups=target_groups, underfilled=len(kept)<target_groups,
        filter_rate=filtered/max(sampled,1), mixed_rate=mixed_sampled/max(sampled,1))
    return GroupSelection(
        indices=flat,
        stats=stats,
        uids=[uids[grouped[index][0]] for index in kept], unavailable_groups=unavailable,
        mixed_groups=mixed_sampled,
    )


def count_infra_groups(uids: Sequence[Hashable], tool_extra_fields: Sequence[Any] | None) -> int:
    """Count groups with at least one infrastructure error.

    Infra groups are counted separately.  They must not be treated as useful
    mixed groups or silently converted into "model failure".
    """
    if tool_extra_fields is None:
        return 0
    if len(uids) != len(tool_extra_fields):
        raise RLError("uids and tool_extra_fields must have the same length")
    bad: set[Hashable] = set()
    for uid, fields in zip(uids, tool_extra_fields):
        if isinstance(fields, dict) and fields.get("infrastructure_error"):
            key = uid.item() if hasattr(uid, "item") else uid
            bad.add(key)
    return len(bad)


def ds_metrics(selection: GroupSelection, *, num_gen_batches: int,
               generated_tokens: int, infra_groups: int,
               optimizer_steps: int, underfilled_batches: int) -> dict[str, Any]:
    """Flatten selection stats into logger-friendly primitives."""
    metrics: dict[str, Any] = {
        "ds/groups_sampled": selection.stats.groups_sampled,
        "ds/groups_kept": selection.stats.groups_kept,
        "ds/groups_filtered": selection.stats.groups_filtered,
        "ds/groups_not_used": selection.stats.groups_not_used,
        "ds/all_zero_groups": selection.stats.all_zero,
        "ds/all_correct_groups": selection.stats.all_correct,
        "ds/constant_other_groups": selection.stats.constant_other,
        "ds/filter_rate": selection.stats.filter_rate,
        "ds/mixed_rate": selection.stats.mixed_rate,
        "ds/mixed_groups": selection.mixed_groups,
        "ds/target_groups": selection.stats.target_groups,
        "ds/underfilled": bool(selection.stats.underfilled),
        "ds/effective_rollouts": len(selection.indices),
        "ds/num_gen_batches": int(num_gen_batches),
        "ds/generated_tokens": int(generated_tokens),
        "ds/infra_groups": int(infra_groups),
        "ds/reward_unavailable_groups": selection.unavailable_groups,
        "ds/optimizer_steps": int(optimizer_steps),
        "ds/underfilled_batches": int(underfilled_batches),
    }
    return metrics
