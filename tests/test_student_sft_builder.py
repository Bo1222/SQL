import json
from pathlib import Path

from e1_alignment.provenance import build_provenance
from scripts.build_student_sft_dataset import (
    build_record,
    classify,
    norm_sql,
)


def provenance():
    return build_provenance(
        source="raw_success",
        generation_mode="n4_sample",
        model_path="artifacts/checkpoints/step_600_merged",
        model_fingerprint="m" * 64,
        runtime_fingerprint="r" * 64,
        retrieval_version="E1.1",
        prompt_mode="query",
        observation_mode="compact",
        split="sft_train",
        final_ex=True,
        final_ex_source="gold_scoring",
    )


def rollout(final_ex=True):
    return {
        "rollout_id": "q1::n4::s0",
        "question_id": "q1",
        "db_id": "db1",
        "final_ex": final_ex,
        "final_sql": "SELECT 1",
        "lookup_empty": 0,
        "lookup_error": 0,
        "sql_error": 1,
        "sample_index": 0,
        "provenance": provenance(),
        "score": {"correct": final_ex, "ok": True},
        "record": {
            "db_path": "db1.sqlite",
            "gold_sql": "SELECT 1",
            "finish_reason": "answer_submitted",
            "invalid_actions": 0,
            "infrastructure_error": False,
            "generation_error": None,
            "final_sql": "SELECT 1",
            "segments": [
                {"kind": "prompt", "ids": [1, 2], "text": "prompt"},
                {"kind": "action", "ids": [3], "text": "<schema_lookup>x</schema_lookup>",
                 "action": "schema_lookup", "arguments": {"query": "x"}, "tool_event_index": 0},
                {"kind": "observation", "ids": [4], "text": "obs",
                 "tool_event_index": 0},
                {"kind": "action", "ids": [5], "text": "<sql_execute>bad</sql_execute>",
                 "action": "sql_execute", "arguments": "bad", "tool_event_index": 1},
                {"kind": "observation", "ids": [6], "text": "error",
                 "tool_event_index": 1},
                {"kind": "action", "ids": [7, 8], "text": "<answer>SELECT 1</answer>",
                 "action": "answer", "arguments": "SELECT 1"},
            ],
            "tool_events": [
                {"action": "schema_lookup", "arguments": {"query": "x"},
                 "result": {"status": "ok", "retrieval_metadata": {"matched_tables": ["t1"]}}},
                {"action": "sql_execute", "arguments": "bad",
                 "result": {"status": "error", "error_type": "schema_error"}},
            ],
        },
    }


def task():
    return {"question_id": "q1", "db_id": "db1", "split": "sft",
            "db_path": "db1.sqlite"}


def test_build_record_masks_failed_action():
    record, reason = build_record(rollout(), task(), max_context=100)
    assert reason is None
    mask = record["trajectory"]["loss_mask"]
    # prompt 2, lookup 1, obs 1, failed sql 1, obs 1, answer 2
    assert mask == [0, 0, 1, 0, 0, 0, 1, 1]
    assert record["trajectory"]["segments"][3]["supervise"] is False
    assert record["trajectory"]["segments"][-1]["supervise"] is True


def test_build_record_rejects_answer_mismatch():
    item = rollout()
    item["record"]["segments"][-1]["arguments"] = "SELECT 2"
    record, reason = build_record(item, task(), max_context=100)
    assert record is None
    assert reason == "answer_sql_mismatch"


def test_build_record_rejects_gold_in_observation():
    item = rollout()
    item["record"]["segments"][2]["text"] = "SELECT 1"
    record, reason = build_record(item, task(), max_context=100)
    assert record is None
    assert reason == "gold_in_prompt_or_observation"


def test_build_record_rejects_over_context():
    record, reason = build_record(rollout(), task(), max_context=5)
    assert record is None
    assert reason == "over_context"


def test_classify_rejects_invalid_action():
    item = rollout()
    item["record"]["invalid_actions"] = 1
    assert classify(item, task(), set(), set()) == "invalid_action"


def test_classify_rejects_eval_overlap():
    assert classify(rollout(), task(), {"q1"}, set()) == "eval_question_overlap"
    assert classify(rollout(), task(), set(), {"db1"}) == "eval_db_overlap"


def test_norm_sql():
    assert norm_sql(" SELECT  1 ; ") == "select 1"
