"""Tests for verifiable process signals and the Stage 0 reward audit."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward.process_signals import (  # noqa: E402
    ProcessEvents,
    compose_reward,
    extract_process_events,
    progress_score,
)
from scripts.audit_process_rewards import (  # noqa: E402
    select_recommendation,
    simulate_groups,
)


def tool(action, *, status="ok", error_type=None, observation=None, arguments=None):
    result = {"status": status}
    if error_type:
        result["error_type"] = error_type
    if observation is not None:
        result["observation"] = observation
    return {"action": action, "arguments": arguments, "result": result}


def test_schema_error_then_lookup_and_repair():
    events = extract_process_events(
        tool_events=[
            tool("sql_execute", status="error", error_type="schema_error", arguments="SELECT bad"),
            tool("schema_lookup", observation="table customers: id, name", arguments={"query": "customers"}),
            tool("sql_execute", arguments="SELECT id FROM customers"),
        ],
        finish_reason="answer_submitted",
        final_sql="SELECT id FROM customers",
        first_sql="SELECT bad",
        invalid_actions=0,
        turns=3,
    )
    assert events.schema_error_count == 1
    assert events.schema_error_then_lookup is True
    assert events.schema_error_then_answer is False
    assert events.schema_error_repaired is True
    assert events.lookup_nonempty is True
    assert events.valid_sql_count == 1
    assert events.answer_submitted is True


def test_schema_error_then_answer_is_flagged():
    events = extract_process_events(
        tool_events=[
            tool("sql_execute", status="error", error_type="schema_error", arguments="SELECT bad"),
        ],
        finish_reason="answer_submitted",
        final_sql="SELECT bad",
        first_sql="SELECT bad",
    )
    assert events.schema_error_then_answer is True
    assert events.schema_error_repaired is False


def test_repeat_and_invalid_counts():
    events = extract_process_events(
        tool_events=[
            tool("sql_execute", arguments="SELECT  1"),
            tool("sql_execute", arguments="select 1"),
            tool("schema_lookup", arguments={"query": "t"}),
            tool("schema_lookup", arguments={"query": "t"}),
        ],
        finish_reason="turn_limit",
        final_sql=None,
        invalid_actions=2,
        turns=4,
    )
    assert events.repeat_sql == 1
    assert events.repeat_lookup == 1
    assert events.invalid_actions == 2
    assert events.answer_submitted is False


def test_progress_scores_are_bounded_and_recovery_is_not_execution_success():
    repaired = extract_process_events(
        tool_events=[
            tool("sql_execute", status="error", error_type="schema_error"),
            tool("sql_execute"),
        ],
        finish_reason="answer_submitted",
        final_sql="SELECT 1",
    )
    not_repaired = extract_process_events(
        tool_events=[tool("sql_execute")],
        finish_reason="answer_submitted",
        final_sql="SELECT 1",
    )
    assert progress_score(repaired, "recovery_only") == 1.0
    assert progress_score(not_repaired, "recovery_only") == 0.0
    for mode in (
        "recovery_only",
        "recovery_penalties",
        "penalties_only",
        "avoid_abandon_only",
        "safe_penalties",
    ):
        value = progress_score(repaired, mode)
        assert 0.0 <= value <= 1.0


def test_abandon_penalty_targets_answer_after_schema_error():
    abandoned = extract_process_events(
        tool_events=[
            tool("sql_execute", status="error", error_type="schema_error"),
        ],
        finish_reason="answer_submitted",
        final_sql="SELECT bad",
    )
    not_abandoned = extract_process_events(
        tool_events=[
            tool("sql_execute", status="error", error_type="schema_error"),
            tool("schema_lookup", observation="table t"),
        ],
        finish_reason="answer_submitted",
        final_sql="SELECT 1",
    )
    assert progress_score(abandoned, "avoid_abandon_only") == 0.0
    assert progress_score(not_abandoned, "avoid_abandon_only") == 1.0
    assert progress_score(abandoned, "safe_penalties") < progress_score(
        not_abandoned, "safe_penalties"
    )


def test_compose_reward_is_rank_preserving():
    events = extract_process_events(tool_events=[], finish_reason="turn_limit", final_sql=None)
    for beta in (0.02, 0.05, 0.1):
        success = compose_reward(True, events, beta=beta)
        failure = compose_reward(False, events, beta=beta)
        assert success == 1.0
        assert -1.0 <= failure <= -0.9
        assert success > failure


def test_compose_reward_rejects_negative_beta():
    events = extract_process_events(tool_events=[], finish_reason="turn_limit", final_sql=None)
    with pytest.raises(ValueError):
        compose_reward(False, events, beta=-0.1)


def _row(ex: bool, repaired: bool, valid_wrong: bool = False):
    if repaired:
        tool_events = [
            tool("sql_execute", status="error", error_type="schema_error"),
            tool("sql_execute"),
        ]
    else:
        tool_events = [tool("sql_execute")]
    events = extract_process_events(
        tool_events=tool_events,
        finish_reason="answer_submitted",
        final_sql="SELECT 1",
    )
    return {
        "question_id": "q",
        "ex": ex,
        "valid": bool(ex or valid_wrong),
        "valid_but_wrong": valid_wrong,
        "infra": False,
        "events": events,
    }


def test_simulate_groups_separates_terminal_and_process_signal():
    groups = [
        [_row(False, False), _row(False, True), _row(False, False), _row(False, True)],
        [_row(True, False), _row(False, True), _row(True, False), _row(False, True)],
        [_row(True, False), _row(True, True), _row(True, False), _row(True, True)],
        [_row(False, False), _row(False, False), _row(False, False), _row(False, False)],
    ]
    result = simulate_groups(groups, betas=[0.05])
    stats = result["recovery_only"]["0.05"]
    assert stats["groups"] == 4
    assert stats["terminal_mixed"] == 1
    assert stats["all_zero_groups"] == 2
    assert stats["all_zero_shaped_mixed"] == 1
    assert stats["all_correct_shaped_mixed"] == 0
    assert stats["process_only_mixed"] == 1


def test_select_recommendation_no_variance_is_no_go():
    signals = {"schema_error_repaired": {"ex_delta": 0.1}}
    simulations = {"recovery_only": {"0.05": {
        "all_zero_groups": 2,
        "all_zero_shaped_mixed": 0,
        "all_zero_shaped_mixed_rate": 0.0,
        "all_zero_best_fail_valid_wrong_rate": None,
        "process_only_mixed_rate": 0.0,
    }}}
    assert select_recommendation(signals, simulations)["decision"] == "no_go"


def test_select_recommendation_conditional_go():
    signals = {"schema_error_repaired": {"ex_delta": 0.1}}
    simulations = {"recovery_only": {"0.05": {
        "all_zero_groups": 10,
        "all_zero_shaped_mixed": 4,
        "all_zero_shaped_mixed_rate": 0.4,
        "all_zero_valid_wrong_trajectory_rate": 0.5,
        "all_zero_best_fail_valid_wrong_rate": 0.3,
        "all_zero_best_fail_valid_wrong_lift": -0.2,
        "process_only_mixed_rate": 0.4,
    }}}
    assert select_recommendation(signals, simulations)["decision"] == "conditional_go"


def test_select_recommendation_no_lift_is_no_go():
    signals = {"schema_error_repaired": {"ex_delta": 0.1}}
    simulations = {"penalties_only": {"0.05": {
        "all_zero_groups": 10,
        "all_zero_shaped_mixed": 4,
        "all_zero_shaped_mixed_rate": 0.4,
        "all_zero_valid_wrong_trajectory_rate": 0.5,
        "all_zero_best_fail_valid_wrong_rate": 0.5,
        "all_zero_best_fail_valid_wrong_lift": 0.0,
        "process_only_mixed_rate": 0.4,
    }}}
    assert select_recommendation(signals, simulations)["decision"] == "no_go"


def test_select_recommendation_rejects_negative_recovery_association():
    signals = {"schema_error_repaired": {"ex_delta": -0.1}}
    simulations = {"recovery_only": {"0.05": {
        "all_zero_groups": 10,
        "all_zero_shaped_mixed": 4,
        "all_zero_shaped_mixed_rate": 0.4,
        "all_zero_valid_wrong_trajectory_rate": 0.5,
        "all_zero_best_fail_valid_wrong_rate": 0.3,
        "all_zero_best_fail_valid_wrong_lift": -0.2,
        "process_only_mixed_rate": 0.4,
    }}}
    assert select_recommendation(signals, simulations)["decision"] == "no_go"
