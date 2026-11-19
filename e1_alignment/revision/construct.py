"""Append real tools and correct suffixes after an immutable failure boundary."""
import copy
import json
from agent.protocol import parse_action
from e1_alignment.common import canonical_hash
from e1_alignment.runtime import process_generation
from e1_alignment.dataset import apply_loss_mask, finalize_record
from e1_alignment.rebuild.ast_check import extract, sqlite_schema
from e1_alignment.rebuild.visibility import coverage
from e1_alignment.rebuild.quality import failure_prefix_check
from e1_alignment.collect_c import missing_query


def action(state, name, arguments):
    c = state.contract
    if state.done or state.rounds >= c.max_rounds:
        raise ValueError("round_budget_exhausted")
    if name != "answer":
        if state.tool_calls >= c.max_tools:
            raise ValueError("tool_budget_exhausted")
        if name == "sql_execute" and state.sql_executes >= c.max_sql_executes:
            raise ValueError("sql_budget_exhausted")
        if name == "schema_lookup" and state.schema_lookups >= c.max_schema_lookups:
            raise ValueError("lookup_budget_exhausted")
    body = json.dumps(arguments, ensure_ascii=False) if name == "schema_lookup" else arguments
    text = f"<{name}>{body}</{name}>"
    ids = state.runtime.tokenizer.encode(text, add_special_tokens=False)
    eos = state.runtime.tokenizer.eos_token_id
    if eos is not None:
        ids.append(eos)
    if len(ids) > c.max_action_tokens or state.context_length() + len(ids) > c.max_context:
        raise ValueError("action_token_budget_exhausted")
    process_generation(state, ids, state.rounds)
    if state.infrastructure_error or state.finish_reason not in (None, "answer_submitted"):
        raise ValueError("real_tool_or_context_failure:" + str(state.finish_reason))


def score_exact(record, score_fn):
    final = score_fn(record["db_path"], record["final_sql"], record["gold_sql"])
    record["final_correct"] = bool(final.get("ok") and final.get("correct") and not final.get("infrastructure_error"))
    record["final_valid"] = bool(final.get("ok"))
    record["final_infrastructure_error"] = bool(final.get("infrastructure_error"))
    for s in record["segments"]:
        if s.get("action") != "sql_execute":
            continue
        scored = score_fn(record["db_path"], s["arguments"], record["gold_sql"])
        s.update(offline_ok=bool(scored.get("ok")), offline_ex=bool(scored.get("correct")),
                 offline_error_type=scored.get("error_type"))
    return final


def build(row, runtime, score_fn, fingerprint, prefix_record=None, boundary=0, target_sql=None):
    """Returns every attempted action on failure as well as successful records."""
    prefix = copy.deepcopy(prefix_record["segments"][1:1 + 2 * boundary]) if boundary else []
    events = copy.deepcopy(prefix_record["tool_events"][:boundary]) if boundary else []
    state = runtime.new_state(row, prompt=prefix_record["prompt"] if boundary else None,
                              prompt_ids=prefix_record["segments"][0]["ids"] if boundary else None,
                              prefix_segments=prefix, tool_events=events)
    try:
        if boundary:
            failed = failure_prefix_check({**prefix_record, "prefix_action_segments": boundary}, score_fn)
            if not failed["ok"]:
                raise ValueError(failed["reason"])
            last = events[-1]["result"]
            if last.get("failure_origin") == "infrastructure" or last.get("error_type") in ("budget", "timeout", "scorer_failure"):
                raise ValueError("not_a_policy_sql_failure")
        sql = target_sql or row["gold_sql"]
        target_score = score_fn(row["db_path"], sql, row["gold_sql"])
        if not target_score.get("ok") or not target_score.get("correct") or target_score.get("infrastructure_error"):
            raise ValueError("target_or_gold_execution_failed")
        schema = sqlite_schema(row["db_path"])
        req = extract(sql, schema=schema)
        if not req["ok"]:
            raise ValueError("target_ast_unresolved")
        # Reserve remaining rounds/tools for one correct execute and one answer.
        while True:
            cov = coverage(req, state.segments, len(state.segments), schema)
            seen_lookup = any(s.get("action") == "schema_lookup" for s in state.segments)
            if seen_lookup and cov["ok"]:
                break
            c = state.contract
            if state.rounds + 3 > c.max_rounds or state.tool_calls + 2 > c.max_tools:
                raise ValueError("insufficient_remaining_repair_budget")
            query = missing_query(row, cov["missing"]) if seen_lookup else (row["question"] + " " + row.get("evidence", "")).strip()
            action(state, "schema_lookup", {"query": query})
        action(state, "sql_execute", sql)
        if state.tool_events[-1]["result"].get("status") != "ok":
            raise ValueError("correct_sql_real_execution_failed")
        action(state, "answer", sql)
        record = state.record()
        record["prefix_action_segments"] = boundary
        score_exact(record, score_fn)
        apply_loss_mask(record, boundary)
        finalize_record(record, fingerprint, "e1_gold_assisted", "programmatic_gold_ast",
                        gold_used_for_construction=True, autonomous_repair=False)
        record["success_type"] = "gold_assisted"
        record["construction_source"] = "gold_assisted"
        record["start_mode"] = "failure_continuation" if boundary else "question_start"
        record["target_sql_basis"] = "reviewed_candidate" if target_sql else "original_gold_pending_semantic_review"
        if boundary:
            record["failure_state"] = {"state_id": "m6-08k:" + canonical_hash([prefix_record["id"], prefix, events]),
                                        "boundary_action_count": boundary}
        return record, {"reason": None, "attempted_record": state.record()}
    except (ValueError, OSError, KeyError) as exc:
        return None, {"reason": str(exc), "attempted_record": state.record()}
