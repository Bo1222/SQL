"""Trajectory scoring, loss masks and dataset records shared by E1-02/03/04/05."""
import json
from pathlib import Path

from e1_alignment.common import normalize_sql, score_sql_cached


def collect_scoring_items(record):
    items = []
    if record.get("final_sql"):
        items.append((record["db_path"], record["final_sql"], record["gold_sql"]))
    for segment in record["segments"]:
        if segment.get("kind") == "action" and segment.get("action") == "sql_execute":
            items.append((record["db_path"], segment.get("arguments"), record["gold_sql"]))
    return items


def attach_scores(record, scores):
    """Attach offline EX to the final answer and every sql_execute action."""
    def lookup(sql):
        key = (str(record["db_path"]), normalize_sql(sql), normalize_sql(record["gold_sql"]))
        return scores.get(key, {"correct": False, "ok": False, "error": "unscored",
                                "infrastructure_error": True})

    final = lookup(record.get("final_sql")) if record.get("final_sql") else {
        "correct": False, "ok": False, "error": "missing_answer", "infrastructure_error": False}
    record["final_correct"] = bool(final.get("correct"))
    record["final_valid"] = bool(final.get("ok"))
    record["final_error_type"] = final.get("error_type")
    record["final_infrastructure_error"] = bool(final.get("infrastructure_error"))
    for segment in record["segments"]:
        if segment.get("kind") == "action" and segment.get("action") == "sql_execute":
            scored = lookup(segment.get("arguments"))
            segment["offline_ex"] = bool(scored.get("correct"))
            segment["offline_ok"] = bool(scored.get("ok"))
            segment["offline_error_type"] = scored.get("error_type")
    return record


def schema_lookup_useful(record, segment):
    """A lookup action is supervised only when the real backend returned context."""
    index = segment.get("tool_event_index")
    if index is None or index >= len(record.get("tool_events", [])):
        return False
    result = record["tool_events"][index].get("result") or {}
    return result.get("status") == "ok" and bool(result.get("observation"))


def apply_loss_mask(record, prefix_action_segments=0):
    """Per-assistant-segment loss mask; observations and failed prefixes stay 0."""
    input_ids, loss_mask = [], []
    action_seen = 0
    for segment in record["segments"]:
        ids = list(segment.get("ids") or [])
        if segment.get("kind") != "action":
            mask = [0] * len(ids)
        else:
            in_prefix = action_seen < prefix_action_segments
            action_seen += 1
            if in_prefix or segment.get("invalid"):
                supervised = False
            elif segment.get("action") == "answer":
                supervised = bool(record.get("final_correct"))
            elif segment.get("action") == "schema_lookup":
                supervised = schema_lookup_useful(record, segment)
            elif segment.get("action") == "sql_execute":
                supervised = bool(segment.get("offline_ex"))
            else:
                supervised = False
            mask = [1 if supervised else 0] * len(ids)
        segment["loss_mask"] = mask
        input_ids += ids
        loss_mask += mask
    record["input_ids"] = input_ids
    record["loss_mask"] = loss_mask
    record["supervised_tokens"] = sum(loss_mask)
    record["supervised_actions"] = sum(
        1 for segment in record["segments"]
        if segment.get("kind") == "action" and any(segment.get("loss_mask", [])))
    return record


def messages(record):
    rows = []
    for segment in record["segments"]:
        kind = segment.get("kind")
        role = "user" if kind == "prompt" else ("assistant" if kind == "action" else "tool")
        rows.append({"role": role, "content": segment.get("text", "")})
    return rows


def tool_calls(record):
    return [{"action": event.get("action"), "arguments": event.get("arguments"),
             "result": event.get("result")} for event in record.get("tool_events", [])]


def observations(record):
    return [segment.get("text", "") for segment in record["segments"] if segment.get("kind") == "observation"]


def classify_success(record):
    """Name the positive trajectory shape (only valid when final_correct)."""
    if not record.get("final_correct"):
        return "failure"
    error_seen = any(
        segment.get("action") == "sql_execute" and segment.get("offline_error_type") not in (None, "none")
        for segment in record["segments"])
    wrong_executable = any(
        segment.get("action") == "sql_execute" and segment.get("offline_ok") and not segment.get("offline_ex")
        for segment in record["segments"])
    lookups = record.get("schema_lookups", 0)
    if error_seen:
        return "autonomous_repair"
    if wrong_executable:
        return "self_correction"
    if lookups:
        return "lookup_success"
    return "direct_sql_success"


def finalize_record(record, fingerprint_sha256, trajectory_source, policy_checkpoint,
                    gold_used_for_construction=False, autonomous_repair=None, extra=None):
    record.update({
        "question_id": record.get("id"),
        "trajectory_source": trajectory_source,
        "policy_checkpoint": policy_checkpoint,
        "runtime_version": "E1",
        "runtime_fingerprint_sha256": fingerprint_sha256,
        "gold_used_for_construction": bool(gold_used_for_construction),
        "autonomous_repair": (classify_success(record) == "autonomous_repair"
                              if autonomous_repair is None else bool(autonomous_repair)),
        "success_type": classify_success(record),
        "messages": messages(record),
        "observations": observations(record),
        "tool_calls": tool_calls(record),
        "input_tokens": sum(len(segment.get("ids") or []) for segment in record["segments"]),
    })
    record.setdefault("excluded_reason", None)
    if extra:
        record.update(extra)
    return record


def is_positive(record):
    """A complete positive trajectory for SFT (A/B); C is validated separately."""
    return (record.get("final_correct") is True
            and record.get("finish_reason") == "answer_submitted"
            and not record.get("infrastructure_error")
            and not record.get("final_infrastructure_error")
            and record.get("invalid_actions", 0) == 0)


def exclusion_reason(record):
    if record.get("infrastructure_error") or record.get("final_infrastructure_error"):
        return "infrastructure_error"
    if record.get("invalid_actions", 0):
        return "invalid_action"
    if record.get("finish_reason") != "answer_submitted":
        return f"finish_{record.get('finish_reason')}"
    if not record.get("final_correct"):
        return "final_ex_zero"
    return None


def score_records(records, workers=8):
    items = []
    for record in records:
        items += collect_scoring_items(record)
    scores = score_sql_cached(items, workers=workers)
    for record in records:
        attach_scores(record, scores)
    return records


def dedupe_key(record):
    actions = [(segment.get("action"), normalize_sql(segment.get("arguments") or ""))
               for segment in record["segments"] if segment.get("kind") == "action"]
    return json.dumps({"source": record.get("trajectory_source"), "question_id": record.get("id"),
                       "actions": actions, "final_sql": normalize_sql(record.get("final_sql") or "")},
                      ensure_ascii=False, sort_keys=True)
