"""CPU tests for the Phase 1A external stop monitor."""
from types import SimpleNamespace

from scripts.phase1a_stop_monitor import evaluate, exit_code_for, summarize


def args(**kwargs):
    base = dict(samples=32, seen_questions=8, total_questions=11,
                min_correct_total=4, max_infra=2)
    base.update(kwargs)
    return SimpleNamespace(**base)


def records_for(question_count, samples=32, correct_questions=(), unavailable=0,
                infra=0):
    rows = []
    for question_index in range(question_count):
        for sample in range(samples):
            rows.append({
                "id": f"q{question_index}", "task_index": question_index,
                "sample_index": sample,
                "correct": question_index in correct_questions and sample == 0,
                "reward_available": sample >= unavailable,
                "infrastructure_error": sample < infra,
            })
    return rows


def test_zero_correct_after_eight_complete_questions_stops():
    reason, complete, solved = evaluate(
        summarize(records_for(8, correct_questions=())), args())
    assert reason and "zero correct" in reason
    assert len(complete) == 8 and solved == 0


def test_seven_complete_zero_correct_does_not_stop():
    reason, complete, solved = evaluate(
        summarize(records_for(7, correct_questions=())), args())
    assert reason is None
    assert len(complete) == 7


def test_infra_over_two_stops():
    reason, _, _ = evaluate(
        summarize(records_for(2, correct_questions=(0,), infra=3)), args())
    assert reason and "infrastructure" in reason


def test_low_yield_at_completion():
    reason, complete, solved = evaluate(
        summarize(records_for(11, correct_questions=(0, 1, 2))), args())
    assert reason and "low yield" in reason
    assert solved == 3


def test_four_solved_at_completion_passes():
    reason, complete, solved = evaluate(
        summarize(records_for(11, correct_questions=(0, 1, 2, 3))), args())
    assert reason is None
    assert solved == 4


def test_exit_code_classifies_low_yield_and_hard_stop():
    assert exit_code_for("low yield: 3/11 questions solved < 4") == 3
    assert exit_code_for("infrastructure errors 3 > 2") == 2
    assert exit_code_for("no new records for 900s") == 2


def test_first_eight_uses_task_index_not_completion_order():
    # Later task indices finish first; the first-8 rule must not use them.
    late = [
        {"id": f"q{index}", "task_index": index, "sample_index": sample,
         "correct": False, "reward_available": True, "infrastructure_error": False}
        for index in range(8, 16) for sample in range(32)
    ]
    early_partial = [
        {"id": f"q{index}", "task_index": index, "sample_index": 0,
         "correct": False, "reward_available": True, "infrastructure_error": False}
        for index in range(8)
    ]
    reason, complete, _ = evaluate(summarize(late + early_partial), args())
    assert reason is None, "later completed questions must not trigger the first-8 rule"
    early_complete = [
        {"id": f"q{index}", "task_index": index, "sample_index": sample,
         "correct": False, "reward_available": True, "infrastructure_error": False}
        for index in range(8) for sample in range(1, 32)
    ]
    reason, complete, _ = evaluate(summarize(late + early_partial + early_complete), args())
    assert reason and "task-index" in reason
    assert len(complete) == 16
