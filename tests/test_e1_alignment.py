"""Contract tests for the M6-08i alignment data pipeline (CPU only)."""
import json

from e1_alignment.collect_b import classify_sql_error
from e1_alignment.collect_c import metadata_coverage
from e1_alignment.dataset import apply_loss_mask, classify_success, dedupe_key, is_positive
from e1_alignment.validate import cross_dedupe_key


def _segment(kind, ids, text, **extra):
    return {"kind": kind, "ids": list(ids), "text": text, **extra}


def _record(final_correct=True, prefix_actions=0):
    segments = [
        _segment("prompt", [1, 2, 3], "prompt"),
        _segment("action", [10, 11], "<schema_lookup>{\"query\": \"x\"}</schema_lookup>",
                 action="schema_lookup", arguments={"query": "x"}, invalid=False, tool_event_index=0),
        _segment("observation", [20, 21], "<observation>ctx</observation>", tool_event_index=0),
        _segment("action", [30], "<sql_execute>SELECT 1</sql_execute>", action="sql_execute",
                 arguments="SELECT 1", invalid=False, tool_event_index=1, offline_ex=False,
                 offline_ok=True, offline_error_type=None),
        _segment("observation", [40], "<observation>rows</observation>", tool_event_index=1),
        _segment("action", [50, 51], "<answer>SELECT 1</answer>", action="answer",
                 arguments="SELECT 1", invalid=False),
    ]
    return {
        "id": "bird_train_x", "db_id": "address", "db_path": "x.sqlite",
        "gold_sql": "SELECT 1", "final_sql": "SELECT 1", "final_correct": final_correct,
        "finish_reason": "answer_submitted", "infrastructure_error": False,
        "invalid_actions": 0, "schema_lookups": 1, "sql_executes": 1, "tool_calls": 2,
        "segments": segments,
        "tool_events": [
            {"action": "schema_lookup", "arguments": {"query": "x"},
             "result": {"status": "ok", "observation": "Relevant Database Context"}},
            {"action": "sql_execute", "arguments": "SELECT 1",
             "result": {"status": "ok", "rows_preview": [[1]]}},
        ],
        "prefix_action_segments": prefix_actions,
    }


def test_segment_mask_only_supervises_useful_actions():
    record = _record()
    apply_loss_mask(record)
    masks = {segment.get("action"): segment["loss_mask"]
             for segment in record["segments"] if segment["kind"] == "action"}
    assert masks["schema_lookup"] == [1, 1]
    assert masks["sql_execute"] == [0]          # executable but not gold-equivalent
    assert masks["answer"] == [1, 1]
    for segment in record["segments"]:
        if segment["kind"] != "action":
            assert not any(segment["loss_mask"])
    assert len(record["input_ids"]) == len(record["loss_mask"])


def test_failed_prefix_is_not_supervised():
    record = _record()
    record["segments"][1]["loss_mask"] = []  # rebuilt by apply_loss_mask
    apply_loss_mask(record, prefix_action_segments=2)
    actions = [segment for segment in record["segments"] if segment["kind"] == "action"]
    assert not any(actions[0]["loss_mask"])
    assert not any(actions[1]["loss_mask"])
    assert any(actions[2]["loss_mask"])


def test_invalid_action_not_supervised():
    record = _record()
    record["segments"][1]["invalid"] = True
    apply_loss_mask(record)
    assert not any(record["segments"][1]["loss_mask"])


def test_success_classification():
    record = _record()
    assert classify_success(record) == "self_correction"
    record["segments"][3]["offline_error_type"] = "syntax_error"
    assert classify_success(record) == "autonomous_repair"
    record["final_correct"] = False
    assert classify_success(record) == "failure"


def test_positive_filter():
    record = _record()
    record["final_correct"] = True
    assert is_positive(record)
    record["invalid_actions"] = 1
    assert not is_positive(record)


def test_dedupe_keys_are_stable():
    left, right = _record(), _record()
    assert dedupe_key(left) == dedupe_key(right)
    assert cross_dedupe_key(left) == cross_dedupe_key(right)
    right["final_sql"] = "SELECT 2"
    assert cross_dedupe_key(left) != cross_dedupe_key(right)


def test_sql_error_classification():
    assert classify_sql_error({"message": "no such table: x"}) == "no_such_table"
    assert classify_sql_error({"message": "no such column: y"}) == "no_such_column"
    assert classify_sql_error({"message": "ambiguous column name: z"}) == "ambiguous_column"
    assert classify_sql_error({"message": "near SELECT: syntax error"}) == "syntax_error"
    assert classify_sql_error({"error_type": "schema_error", "message": "other"}) == "schema_error"


def test_metadata_coverage_requires_tables_columns_and_joins():
    requirements = {"tables": ["a", "b"], "columns": ["a.id", "b.a_id"],
                    "join_keys": ["a.id=b.a_id"]}
    union = {"tables": {"a", "b"},
             "columns": [{"table": "a", "column": "id"}, {"table": "b", "column": "a_id"}],
             "relationships": [{"from_table": "a", "from_column": "id",
                                "to_table": "b", "to_column": "a_id"}],
             "values": []}
    coverage = metadata_coverage(requirements, union)
    assert coverage["tables_complete"] and coverage["columns_complete"] and coverage["joins_complete"]
    union["columns"] = union["columns"][:1]
    coverage = metadata_coverage(requirements, union)
    assert coverage["tables_complete"] and not coverage["columns_complete"]
