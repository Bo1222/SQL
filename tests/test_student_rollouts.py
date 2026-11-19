from types import SimpleNamespace

from scripts.run_student_rollouts import (
    find_repair_prefix,
    lookup_counts,
    norm_sql,
)


def fake_state():
    segments = [
        {"kind": "prompt", "ids": [1], "text": "prompt"},
        {"kind": "action", "ids": [2], "text": "<schema_lookup>city</schema_lookup>",
         "action": "schema_lookup", "arguments": {"query": "city"}, "tool_event_index": 0},
        {"kind": "observation", "ids": [3], "text": "obs0", "tool_event_index": 0},
        {"kind": "action", "ids": [4], "text": "<sql_execute>bad</sql_execute>",
         "action": "sql_execute", "arguments": "bad", "tool_event_index": 1},
        {"kind": "observation", "ids": [5], "text": "obs1", "tool_event_index": 1},
        {"kind": "action", "ids": [6], "text": "<answer>bad</answer>",
         "action": "answer", "arguments": "bad"},
    ]
    tool_events = [
        {"action": "schema_lookup", "arguments": {"query": "city"},
         "result": {"status": "ok", "retrieval_metadata": {"matched_tables": ["city"]}}},
        {"action": "sql_execute", "arguments": "bad",
         "result": {"status": "error", "error_type": "schema_error"}},
    ]
    return SimpleNamespace(segments=segments, tool_events=tool_events)


def test_find_repair_prefix_stops_before_first_sql_error():
    prefix, events, reason = find_repair_prefix(fake_state())
    assert reason == "sql_error"
    assert all(segment["kind"] != "prompt" for segment in prefix)
    assert [segment["kind"] for segment in prefix] == ["action", "observation"]
    assert len(events) == 1
    assert events[0]["action"] == "schema_lookup"


def test_find_repair_prefix_stops_before_invalid_action():
    state = fake_state()
    state.segments[1]["invalid"] = True
    prefix, events, reason = find_repair_prefix(state)
    assert reason == "invalid_action"
    assert prefix == []
    assert events == []


def test_find_repair_prefix_stops_before_wrong_answer():
    state = fake_state()
    state.tool_events[1]["result"] = {"status": "ok"}
    prefix, events, reason = find_repair_prefix(state)
    assert reason == "wrong_answer"
    assert [segment["kind"] for segment in prefix] == ["action", "observation", "action", "observation"]
    assert len(events) == 2


def test_lookup_counts():
    events = [
        {"action": "schema_lookup", "result": {"status": "ok", "retrieval_metadata": {"matched_tables": []}}},
        {"action": "schema_lookup", "result": {"status": "error"}},
        {"action": "sql_execute", "result": {"status": "error"}},
        {"action": "sql_execute", "result": {"status": "ok"}},
    ]
    assert lookup_counts(events) == (1, 1, 1)


def test_norm_sql():
    assert norm_sql("  SELECT  1 ; ") == "select 1"
