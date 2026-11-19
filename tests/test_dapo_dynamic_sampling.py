"""CPU tests for the project-local DAPO dynamic-sampling adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e1_alignment.rl_core import RLError  # noqa: E402
from train.dynamic_sampling import (  # noqa: E402
    count_infra_groups,
    ds_metrics,
    group_indices_by_uid,
    select_group_indices,
)


def test_group_indices_preserve_order():
    assert group_indices_by_uid(["a", "a", "b", "b", "c", "c"]) == [
        [0, 1], [2, 3], [4, 5]
    ]


def test_select_keeps_only_mixed_groups():
    uids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    rewards = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    result = select_group_indices(uids, rewards, target_groups=1, max_groups=4)
    assert result.indices == [4, 5]
    assert result.stats.groups_sampled == 4
    assert result.stats.groups_kept == 1
    assert result.stats.groups_filtered == 3
    assert result.stats.underfilled is False
    assert result.stats.all_zero == 1
    assert result.stats.all_correct == 2


def test_select_underfilled_when_not_enough_mixed():
    uids = ["a", "a", "b", "b", "c", "c"]
    rewards = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    result = select_group_indices(uids, rewards, target_groups=2, max_groups=3)
    assert result.indices == []
    assert result.stats.underfilled is True
    assert result.stats.groups_kept == 0


def test_select_truncates_to_max_groups_budget():
    uids = ["a", "a", "b", "b", "c", "c"]
    rewards = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    result = select_group_indices(uids, rewards, target_groups=1, max_groups=3)
    assert result.indices == [2, 3]
    assert result.stats.groups_not_used == 1
    assert result.stats.mixed_rate == pytest.approx(2 / 3)
    assert result.mixed_groups == 2
    assert result.uids == ["b"]


def test_select_rejects_singleton_groups():
    with pytest.raises(RLError):
        select_group_indices(["a", "b", "b"], [0.0, 0.0, 1.0],
                             target_groups=1, max_groups=2)


def test_select_rejects_inconsistent_group_size():
    with pytest.raises(RLError):
        select_group_indices(["a", "a", "b"], [0.0, 1.0, 1.0],
                             target_groups=1, max_groups=2)


def test_count_infra_groups_is_separate_from_reward():
    uids = ["a", "a", "b", "b", "c", "c"]
    fields = [
        {"infrastructure_error": False}, {"infrastructure_error": False},
        {"infrastructure_error": True}, {"infrastructure_error": False},
        {"infrastructure_error": False}, {"infrastructure_error": False},
    ]
    assert count_infra_groups(uids, fields) == 1
    assert count_infra_groups(uids, None) == 0


def test_unavailable_member_excludes_the_entire_group_and_refills():
    result = select_group_indices(["bad"]*4+["good"]*4, [1,1,1,0,1,0,1,0],
        target_groups=1, max_groups=2, reward_available=[True,True,True,False]+[True]*4)
    assert result.indices == [4,5,6,7]
    assert result.unavailable_groups == 1
    assert result.stats.groups_filtered == 1


def test_vanilla_grpo_keeps_legitimate_constant_groups_but_excludes_unavailable():
    result = select_group_indices(["bad"]*2+["zero"]*2+["one"]*2, [1,0,0,0,1,1],
        target_groups=2, max_groups=3, reward_available=[True,False]+[True]*4, require_mixed=False)
    assert result.indices == [2,3,4,5]
    assert result.stats.all_zero == result.stats.all_correct == 1
    assert result.unavailable_groups == 1


def test_all_unavailable_groups_are_underfilled_not_negative_examples():
    result = select_group_indices(["bad"]*4,[1,1,1,0],target_groups=1,max_groups=1,
                                  reward_available=[True,True,True,False])
    assert result.indices == [] and result.stats.underfilled
    assert result.stats.all_zero == 0


def test_ds_metrics_are_logger_friendly():
    result = select_group_indices(
        ["a", "a", "b", "b"], [0.0, 1.0, 0.0, 0.0],
        target_groups=1, max_groups=2,
    )
    metrics = ds_metrics(
        result, num_gen_batches=1, generated_tokens=16,
        infra_groups=0, optimizer_steps=3, underfilled_batches=1,
    )
    assert metrics["ds/groups_kept"] == 1
    assert metrics["ds/effective_rollouts"] == 2
    assert metrics["ds/filter_rate"] == 0.5
    assert metrics["ds/underfilled_batches"] == 1
    assert metrics["ds/optimizer_steps"] == 3
