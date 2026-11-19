"""Exact source pointers; historical tokens and EOS are never reconstructed."""
import copy
from functools import lru_cache
from pathlib import Path
from e1_alignment.common import read_jsonl, canonical_hash


def pointer(path, line, row):
    return {"file": str(Path(path).resolve()), "line": line, "sha256": canonical_hash(row)}


@lru_cache(maxsize=16)
def _rows(path, mtime_ns):
    return read_jsonl(path)


def resolve(ref):
    path = Path(ref["file"])
    rows = _rows(str(path), path.stat().st_mtime_ns)
    index = ref["line"] - 1
    if not 0 <= index < len(rows) or canonical_hash(rows[index]) != ref["sha256"]:
        raise ValueError("source_receipt_mismatch")
    return rows[index]


def history(segments):
    # Offline scores and masks may change, visible history cannot.
    keys = ("kind", "text", "ids", "action", "arguments", "invalid", "tool_event_index")
    return [{k: s.get(k) for k in keys} for s in segments]


def events(events):
    # Latencies may differ during replay, but source association uses exact originals.
    return [{k: e.get(k) for k in ("action", "arguments", "result")} for e in events]


def match_prefix(record, sources, count, complete=False):
    """Match actual prompt, tokens, EOS, events and input slice, not SQL similarity."""
    wanted = history(record["segments"] if complete else record["segments"][:1 + 2 * count])
    used = events(record.get("tool_events", [])[:count])
    for path, line, raw in sources:
        if raw.get("id", raw.get("question_id")) != record["id"]:
            continue
        if wanted != history(raw.get("segments", [])[:len(wanted)]):
            continue
        if used != events(raw.get("tool_events", [])[:count]):
            continue
        return pointer(path, line, raw)
    return None


def verify_history(record):
    if record.get("construction_source") == "gold_assisted" and not record.get("acquisition_receipt"):
        raise ValueError("gold_construction_requires_real_acquisition_receipt")
    source = resolve(record["history_origin"])
    parent = resolve(record["origin"])
    count = record.get("prefix_action_segments", 0)
    if record.get("start_mode") == "failure_continuation":
        if count <= 0:
            raise ValueError("invalid_failure_boundary")
        stop = 1 + 2 * count
        if history(record["segments"][:stop]) != history(source["segments"][:stop]):
            raise ValueError("historical_prefix_changed")
        if events(record["tool_events"][:count]) != events(source["tool_events"][:count]):
            raise ValueError("historical_feedback_changed")
    elif record.get("construction_source") == "model_generated":
        if history(record["segments"]) != history(source["segments"]):
            raise ValueError("model_history_changed")
        if events(record["tool_events"]) != events(source["tool_events"]):
            raise ValueError("model_feedback_changed")
    for key in ("id", "db_id", "question", "evidence", "gold_sql", "db_path"):
        if (record.get(key) or "") != (source.get(key) or ""):
            raise ValueError("source_task_mismatch:" + key)
        if (record.get(key) or "") != (parent.get(key) or ""):
            raise ValueError("parent_task_mismatch:" + key)
    if record.get("acquisition_receipt"):
        acquired = resolve(record["acquisition_receipt"])
        if history(acquired["segments"]) != history(record["segments"]) or events(acquired["tool_events"]) != events(record["tool_events"]):
            raise ValueError("acquired_history_mismatch")
    return source
