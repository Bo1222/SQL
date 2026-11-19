import math

import pytest
import torch

from e1_alignment.rl_core import (
    RLError,
    RolloutGroup,
    action_spans,
    group_advantages,
    kl_penalty,
    overlong_penalty,
    policy_loss,
    segments_to_response,
    select_mixed_groups,
    trajectory_tensors,
    validate_training_tensors,
)


def test_group_advantages_standard_grpo_population_std():
    advantages, stats = group_advantages(
        [1.0, 0.0, 0.0, 1.0], ["a", "a", "a", "a"], unbiased=False
    )
    assert advantages == pytest.approx([1.0, -1.0, -1.0, 1.0], abs=1e-5)
    assert stats.groups_total == 1
    assert stats.groups_mixed == 1
    assert stats.mixed_rate == 1.0
    assert stats.rewards_mean == pytest.approx(0.5)
    assert stats.rewards_std == pytest.approx(0.5)


def test_group_advantages_verl_sample_std_default():
    # veRL calls torch.std without unbiased=False, i.e. denominator n-1.
    advantages, _ = group_advantages([1.0, 0.0, 0.0, 1.0], ["a"] * 4)
    expected = 0.5 / (1.0 / 3.0) ** 0.5
    assert advantages == pytest.approx([expected, -expected, -expected, expected], abs=1e-5)


def test_group_advantages_constant_group_is_zero_not_nan():
    advantages, stats = group_advantages([0.0, 0.0, 0.0], ["x"] * 3)
    assert advantages == [0.0, 0.0, 0.0]
    assert stats.groups_all_zero == 1
    assert stats.groups_mixed == 0
    assert math.isfinite(stats.rewards_std)


def test_group_advantages_dr_grpo_centering_without_std():
    advantages, _ = group_advantages(
        [2.0, 4.0], ["g", "g"], normalize_by_std=False
    )
    assert advantages == pytest.approx([-1.0, 1.0])


def test_group_advantages_requires_two_rollouts():
    with pytest.raises(RLError):
        group_advantages([1.0], ["single"])


def test_group_advantages_preserves_interleaved_order():
    advantages, _ = group_advantages(
        [1.0, 0.0, 0.0, 1.0], ["a", "b", "a", "b"], unbiased=False
    )
    assert advantages == pytest.approx([1.0, -1.0, -1.0, 1.0], abs=1e-5)


def test_dynamic_sampling_filters_same_groups_and_counts_budget():
    groups = [
        RolloutGroup("zero", [0.0, 0.0, 0.0, 0.0]),
        RolloutGroup("mixed1", [1.0, 0.0, 0.0, 1.0]),
        RolloutGroup("full", [1.0, 1.0, 1.0, 1.0]),
        RolloutGroup("mixed2", [0.0, 0.0, 0.0, 1.0]),
        RolloutGroup("mixed3", [1.0, 1.0, 0.0, 0.0]),
    ]
    result = select_mixed_groups(groups, target_groups=2, max_groups=4)
    assert [g.group_id for g in result.kept] == ["mixed1", "mixed2"]
    stats = result.stats
    assert stats.groups_sampled == 4
    assert stats.groups_kept == 2
    assert stats.groups_filtered == 2
    assert stats.all_zero == 1
    assert stats.all_correct == 1
    assert stats.groups_not_used == 0
    assert stats.underfilled is False
    assert stats.filter_rate == pytest.approx(0.5)


def test_dynamic_sampling_underfilled_budget():
    groups = [
        RolloutGroup("zero", [0.0, 0.0, 0.0, 0.0]),
        RolloutGroup("full", [1.0, 1.0, 1.0, 1.0]),
        RolloutGroup("mixed", [1.0, 0.0, 0.0, 0.0]),
    ]
    result = select_mixed_groups(groups, target_groups=2)
    assert len(result.kept) == 1
    assert result.stats.underfilled is True
    assert result.stats.groups_filtered == 2


def test_policy_loss_ratio_one_is_negative_advantage_mean():
    logp = torch.zeros(1, 4)
    old = torch.zeros(1, 4)
    mask = torch.ones(1, 4)
    adv = torch.full((1, 1), 0.5)
    loss, metrics = policy_loss(logp, old, adv, mask)
    assert loss.item() == pytest.approx(-0.5)
    assert metrics["clipfrac"] == 0.0
    assert metrics["ratio_mean"] == pytest.approx(1.0)


def test_clip_higher_is_asymmetric():
    logp = torch.tensor([[math.log(1.3)]])
    old = torch.zeros(1, 1)
    mask = torch.ones(1, 1)

    positive_loss, positive_metrics = policy_loss(
        logp, old, torch.ones(1, 1), mask, clip_low=0.2, clip_high=0.28
    )
    # max(-1.3, -1.28) = -1.28
    assert positive_loss.item() == pytest.approx(-1.28)
    assert positive_metrics["clipfrac"] == 1.0

    negative_loss, _ = policy_loss(
        logp, old, -torch.ones(1, 1), mask, clip_low=0.2, clip_high=0.28
    )
    # max(1.3, 1.28) = 1.3; the lower clip does not bind on a negative advantage
    assert negative_loss.item() == pytest.approx(1.3)


def test_policy_loss_token_mean_vs_sequence_mean():
    logp = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    adv = torch.tensor([[1.0], [1.0]])
    token_loss, _ = policy_loss(logp, old, adv, mask, agg_mode="token-mean")
    seq_loss, _ = policy_loss(logp, old, adv, mask, agg_mode="seq-mean-token-mean")
    # Token mean: 4 valid tokens with -1 each -> -1.0
    # Sequence mean: both sequences have loss -1 -> -1.0
    assert token_loss.item() == pytest.approx(-1.0)
    assert seq_loss.item() == pytest.approx(-1.0)

    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    logp = torch.zeros(2, 4)
    old = torch.zeros(2, 4)
    token_loss, _ = policy_loss(logp, old, adv, mask, agg_mode="token-mean")
    seq_loss, _ = policy_loss(logp, old, adv, mask, agg_mode="seq-mean-token-mean")
    assert token_loss.item() == pytest.approx(-1.0)
    assert seq_loss.item() == pytest.approx(-1.0)


def test_policy_loss_global_denominator_scales_loss():
    logp = torch.zeros(1, 2)
    old = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    adv = torch.ones(1, 1)
    local, _ = policy_loss(logp, old, adv, mask)
    global_loss, _ = policy_loss(logp, old, adv, mask, global_num_tokens=8)
    assert global_loss.item() == pytest.approx(local.item() / 4.0)


def test_policy_loss_rejects_padding_in_mask():
    with pytest.raises(RLError):
        policy_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 1),
            torch.tensor([[1.0, 2.0]]),
        )


def test_kl_estimators():
    logp = torch.zeros(1, 2)
    ref = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    assert kl_penalty(logp, ref, mask, kind="kl").item() == 0.0
    assert kl_penalty(logp, ref, mask, kind="low_var_kl").item() == 0.0
    ref = torch.full((1, 2), math.log(2.0))
    expected = math.exp(math.log(2.0)) - math.log(2.0) - 1.0
    assert kl_penalty(logp, ref, mask, kind="low_var_kl").item() == pytest.approx(expected)
    # veRL's k1 is policy-minus-reference, so this is negative.
    assert kl_penalty(logp, ref, mask, kind="kl").item() == pytest.approx(-math.log(2.0))


def test_overlong_penalty_matches_dapo_linear_shape():
    assert overlong_penalty(3000, max_response_length=4096, buffer_length=512) == 0.0
    assert overlong_penalty(3584, max_response_length=4096, buffer_length=512) == 0.0
    assert overlong_penalty(3840, max_response_length=4096, buffer_length=512) == pytest.approx(-0.5)
    assert overlong_penalty(4096, max_response_length=4096, buffer_length=512) == pytest.approx(-1.0)
    # DAPORewardManager does not clamp out-of-range inputs; responses are
    # capped at max_response_length by the rollout, so this is a guard only.
    assert overlong_penalty(5000, max_response_length=4096, buffer_length=512) == pytest.approx(-2.765625)


def test_segments_mask_marks_actions_and_not_observations():
    segments = [
        {"kind": "action", "ids": [11, 12]},
        {"kind": "observation", "ids": [21, 22, 23]},
        {"kind": "action", "ids": [31], "invalid": True},
        {"kind": "observation", "ids": [41]},
        {"kind": "action", "ids": [51, 52]},
    ]
    ids, mask = segments_to_response(segments)
    assert ids == [11, 12, 21, 22, 23, 31, 41, 51, 52]
    assert mask == [1, 1, 0, 0, 0, 1, 0, 1, 1]
    assert action_spans(segments) == [(0, 2), (5, 6), (7, 9)]


def test_trajectory_tensors_skips_explicit_prompt_segment():
    record = {
        "prompt_ids": [1, 2, 3],
        "segments": [
            {"kind": "prompt", "ids": [1, 2, 3]},
            {"kind": "action", "ids": [10, 11]},
            {"kind": "observation", "ids": [20]},
            {"kind": "action", "ids": [30]},
        ],
    }
    result = trajectory_tensors(record)
    assert result["input_ids"] == [1, 2, 3, 10, 11, 20, 30]
    assert result["response_mask"] == [1, 1, 0, 1]
    assert action_spans(record["segments"]) == [(0, 2), (3, 4)]


def test_trajectory_tensors_validates_and_builds_input_ids():
    record = {
        "prompt_ids": [1, 2, 3],
        "segments": [
            {"kind": "action", "ids": [10, 11]},
            {"kind": "observation", "ids": [20]},
            {"kind": "action", "ids": [30]},
        ],
    }
    result = trajectory_tensors(record)
    assert result["input_ids"] == [1, 2, 3, 10, 11, 20, 30]
    assert result["response_ids"] == [10, 11, 20, 30]
    assert result["response_mask"] == [1, 1, 0, 1]
    assert result["prompt_length"] == 3
    assert result["response_length"] == 4


def test_validate_training_tensors_rejects_length_mismatch():
    with pytest.raises(RLError):
        validate_training_tensors(
            input_ids=[1, 2, 3, 4],
            response_mask=[1],
            prompt_length=2,
        )
    with pytest.raises(RLError):
        validate_training_tensors(
            input_ids=[1, 2, 3],
            response_mask=[1, 0],
            prompt_length=2,
            old_log_probs=[0.0],
        )
