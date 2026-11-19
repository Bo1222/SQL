"""Programmatic gates and content-bound, per-trajectory semantic review contracts."""
import copy
from pathlib import Path

from agent.protocol import parse_action
from e1_alignment.common import canonical_hash
from e1_alignment.dataset import apply_loss_mask
from e1_alignment.rebuild import RULE_VERSION
from e1_alignment.rebuild.ast_check import extract, sqlite_schema, AST_RULE_VERSION, PARSER_VERSION
from e1_alignment.rebuild.visibility import coverage

SOURCES = {"e1_self_rollout": "A", "e1_repair_continuation": "B", "e1_gold_assisted": "C"}
REVIEW_FIELDS = ("task_alignment", "sql_semantics", "visible_grounding", "recovery",
                 "result_coincidence", "supervision")
DERIVED = {"content_sha256", "program_check", "semantic_review", "ast_check",
           "visible_coverage", "join_verification"}


def content_hash(record):
    return canonical_hash({k: v for k, v in record.items() if k not in DERIVED})


def versioned(record, origin, line):
    row = copy.deepcopy(record)
    row["trajectory_id"] = "m6-08j:" + canonical_hash({"origin": str(origin), "line": line,
                                                       "record": record})[:24]
    row["parent_version_id"] = None
    row["revision"] = 0
    row["origin"] = {"file": str(origin), "line": line, "sha256": canonical_hash(record)}
    row["quality_rule_version"] = RULE_VERSION
    row["content_sha256"] = content_hash(row)
    return row


def revised(parent, replacement, reason, acquisition_receipt=None):
    if parent.get("revision", 0) >= 1:
        raise ValueError("one_revision_limit")
    if any(replacement.get(k) != parent.get(k) for k in ("id", "db_id", "question", "evidence", "gold_sql", "db_path")):
        raise ValueError("cannot_modify_task_or_gold")
    def visible_history(r):
        return [(s.get("kind"), s.get("text"), s.get("ids"), s.get("action"), s.get("arguments"))
                for s in r.get("segments", [])]
    if visible_history(parent) != visible_history(replacement):
        if not acquisition_receipt:
            raise ValueError("action_or_observation_revision_requires_real_acquisition_receipt")
        from e1_alignment.common import read_jsonl
        path = Path(acquisition_receipt["file"]).resolve()
        root = Path("artifacts/m6/sft_rebuild").resolve()
        if not path.is_relative_to(root):
            raise ValueError("revision_acquisition_outside_branch")
        rows = read_jsonl(path)
        index = acquisition_receipt["line"] - 1
        if not 0 <= index < len(rows) or canonical_hash(rows[index]) != acquisition_receipt["sha256"]:
            raise ValueError("revision_acquisition_receipt_mismatch")
        if visible_history(rows[index]) != visible_history(replacement):
            raise ValueError("revision_not_real_acquired_history")
        if parent.get("trajectory_source") != "e1_gold_assisted" and replacement.get("gold_used_for_construction"):
            if replacement.get("trajectory_source") != "e1_gold_assisted":
                raise ValueError("gold_modified_action_must_be_gold_assisted")
    row = copy.deepcopy(replacement)
    row["acquisition_receipt"] = acquisition_receipt
    row["parent_version_id"] = parent["trajectory_id"]
    row["revision"] = 1
    row["revision_reason"] = reason
    row["revision_diff"] = {k: {"before_hash": canonical_hash(parent.get(k)),
                                 "after_hash": canonical_hash(row.get(k))}
                              for k in set(parent) | set(row)
                              if k not in DERIVED and parent.get(k) != row.get(k)}
    row["trajectory_id"] = "m6-08j:" + canonical_hash([parent["trajectory_id"], row])[:24]
    row["quality_rule_version"] = RULE_VERSION
    for field in DERIVED:
        row.pop(field, None)
    row["content_sha256"] = content_hash(row)
    return row


def sequence_errors(segments, events):
    reasons = []
    if not segments or segments[0].get("kind") != "prompt":
        return ["missing_initial_prompt"]
    position, event_index, answered = 1, 0, False
    while position < len(segments):
        segment = segments[position]
        if segment.get("kind") != "action":
            reasons.append("orphan_or_unexpected_segment")
            position += 1
            continue
        if segment.get("action") == "answer":
            answered = True
            if position != len(segments) - 1:
                reasons.append("answer_not_last")
            break
        if segment.get("tool_event_index") != event_index:
            reasons.append("action_event_index_mismatch")
        following = segments[position + 1] if position + 1 < len(segments) else {}
        if following.get("kind") != "observation" or following.get("tool_event_index") != event_index:
            reasons.append("missing_or_misbound_observation")
        position += 2
        event_index += 1
    if not answered:
        reasons.append("missing_terminal_answer")
    if event_index != len(events):
        reasons.append("tool_event_count_mismatch")
    return sorted(set(reasons))


def failure_prefix_check(record, score_fn=None):
    n = int(record.get("prefix_action_segments", 0))
    actions = [s for s in record.get("segments", []) if s.get("kind") == "action"][:n]
    executes = [s for s in actions if s.get("action") == "sql_execute"]
    if not executes:
        return {"ok": False, "reason": "failure_prefix_has_no_sql"}
    last = executes[-1]
    index = last.get("tool_event_index")
    events = record.get("tool_events", [])
    result = events[index].get("result", {}) if isinstance(index, int) and 0 <= index < len(events) else {}
    if result.get("status") == "error" and result.get("failure_origin") != "infrastructure":
        return {"ok": True, "kind": "real_sql_error"}
    if score_fn is None:
        return {"ok": False, "reason": "failure_prefix_requires_original_sql_scoring"}
    scored = score_fn(record["db_path"], last["arguments"], record["gold_sql"])
    if not scored.get("ok") or scored.get("infrastructure_error"):
        return {"ok": False, "reason": "failure_prefix_score_not_verified", "score": scored}
    if scored.get("correct"):
        return {"ok": False, "reason": "failure_prefix_latest_sql_already_correct", "score": scored}
    return {"ok": True, "kind": "real_executable_semantic_failure", "score": scored}


def program_check(record, train_rows, monitor_ids, expected_fingerprint, runtime=None,
                  score=True, forbidden_dbs=(), prefix_index=None, *,
                  rule_version=RULE_VERSION, failure_continuation=False, score_function=None):
    reasons, warnings = [], []
    row = train_rows.get(record.get("id"))
    if row is None:
        reasons.append("not_in_rl_train")
    else:
        for key in ("question", "evidence", "gold_sql", "db_id", "db_path"):
            if (record.get(key) or "") != (row.get(key) or ""):
                reasons.append(f"train_identity_mismatch:{key}")
    if record.get("id") in monitor_ids:
        reasons.append("monitor_leakage")
    if record.get("db_id") in forbidden_dbs:
        reasons.append("evaluation_database_leakage")
    if record.get("runtime_fingerprint_sha256") != expected_fingerprint:
        reasons.append("runtime_fingerprint_mismatch")
    source = SOURCES.get(record.get("trajectory_source"))
    if source is None:
        reasons.append("unknown_source")
    if (source == "C") != (record.get("gold_used_for_construction") is True):
        reasons.append("gold_source_flag_mismatch")
    if source == "C" and (record.get("success_type") != "gold_assisted" or record.get("autonomous_repair") is not False):
        reasons.append("gold_assisted_marked_autonomous")
    if not record.get("final_correct") or record.get("infrastructure_error") or record.get("final_infrastructure_error"):
        reasons.append("not_complete_positive")
    if record.get("quality_rule_version") != rule_version:
        reasons.append("quality_rule_version_mismatch")
    if content_hash(record) != record.get("content_sha256"):
        reasons.append("content_hash_mismatch")
    segments = record.get("segments") or []
    reasons.extend(sequence_errors(segments, record.get("tool_events") or []))
    if source == "C" and not failure_continuation:
        actions = [s for s in segments if s.get("kind") == "action"]
        initial_query = ((record.get("question") or "") + " " + (record.get("evidence") or "")).strip()
        if not actions or actions[0].get("action") != "schema_lookup" or actions[0].get("arguments") != {"query": initial_query}:
            reasons.append("gold_first_lookup_not_question_evidence")
    flat_ids = [t for s in segments for t in s.get("ids", [])]
    flat_mask = [t for s in segments for t in s.get("loss_mask", [])]
    if flat_ids != record.get("input_ids") or flat_mask != record.get("loss_mask"):
        reasons.append("segment_flatten_mismatch")
    if not flat_ids or len(flat_ids) != len(flat_mask) or any(m not in (0, 1) for m in flat_mask):
        reasons.append("invalid_mask")
    if not any(flat_mask[1:]) or (flat_mask and flat_mask[0]):
        reasons.append("invalid_shifted_supervision")
    expected = copy.deepcopy(record)
    apply_loss_mask(expected, int(record.get("prefix_action_segments", 0)))
    if expected.get("loss_mask") != record.get("loss_mask"):
        reasons.append("supervision_policy_mismatch")
    if source == "B" or failure_continuation:
        n = record.get("prefix_action_segments", 0)
        if not n or not record.get("failure_state"):
            reasons.append("missing_real_failure_boundary")
        if prefix_index is None:
            reasons.append("failure_prefix_not_verified")
        else:
            states = prefix_index.get((record.get("failure_state") or {}).get("state_id")) or []
            states = [states] if isinstance(states, dict) else states
            prefix = segments[1:1 + 2 * n]
            actual = canonical_hash([(s.get("kind"), s.get("text"), s.get("ids")) for s in prefix])
            if not any(actual == canonical_hash([(s.get("kind"), s.get("text"), s.get("ids"))
                                                  for s in state.get("segments", [])]) for state in states):
                reasons.append("failure_prefix_changed_or_missing")
    checks, asts = [], {}
    try:
        schema = sqlite_schema(str(record["db_path"]))
        asts["gold"] = extract(record.get("gold_sql", ""), schema=schema)
        asts["final"] = extract(record.get("final_sql", ""), schema=schema)
        for label, ast in asts.items():
            if not ast["ok"]:
                reasons.append(f"{label}_ast_unresolved")
        for position, segment in enumerate(segments):
            ids, mask = segment.get("ids", []), segment.get("loss_mask", [])
            if len(ids) != len(mask) or (segment.get("kind") != "action" and any(mask)):
                reasons.append("segment_mask_invalid")
            if segment.get("kind") != "action":
                continue
            try:
                action = parse_action(segment.get("text", ""))
                if action.name != segment.get("action") or action.arguments != segment.get("arguments"):
                    reasons.append("action_metadata_mismatch")
            except (ValueError, TypeError):
                reasons.append("action_parse_error")
            if segment.get("action") not in ("sql_execute", "answer") or not any(mask):
                continue
            requirements = extract(segment["arguments"], schema=schema)
            asts[str(position)] = requirements
            if not requirements["ok"]:
                reasons.append("supervised_sql_ast_unresolved")
                continue
            earlier_lookup = any(s.get("action") == "schema_lookup" for s in segments[:position])
            if source == "C" or earlier_lookup:
                cov = coverage(requirements, segments, position, schema)
                checks.append(cov)
                if not cov["ok"]:
                    reasons.append("visible_schema_incomplete")
        if source == "C":
            # Gold coverage before the FIRST SQL target, never its execution feedback.
            first = next((i for i, s in enumerate(segments) if s.get("action") in ("sql_execute", "answer")
                          and (not failure_continuation or any(s.get("loss_mask", [])))), 0)
            cov = coverage(asts["gold"], segments, first, schema)
            checks.append(cov)
            if not cov["ok"]:
                reasons.append("gold_visible_schema_incomplete")
    except (OSError, ValueError, KeyError) as exc:
        reasons.append(f"schema_unavailable:{exc}")
    prefix_failure = None
    if source == "B" or failure_continuation:
        from reward.bird_ex_reward import score_sql
        score_sql = score_function or score_sql
        prefix_failure = failure_prefix_check(record, score_sql if score else None)
        if not prefix_failure["ok"]:
            reasons.append(prefix_failure["reason"])
    scored = None
    supervised_sql_scores = {}
    if score:
        from reward.bird_ex_reward import score_sql
        score_sql = score_function or score_sql
        scored = score_sql(record.get("db_path", ""), record.get("final_sql"), record.get("gold_sql"), timeout=30)
        if not scored.get("ok") or not scored.get("correct") or scored.get("infrastructure_error"):
            reasons.append("full_execution_or_score_failed")
        # Historical score_records normalized whitespace even inside SQL literals.
        # Recheck every supervised execute using the original SQL and original Gold.
        exact_scores = {record.get("final_sql"): scored}
        for position, segment in enumerate(segments):
            if segment.get("action") != "sql_execute" or not any(segment.get("loss_mask", [])):
                continue
            sql = segment.get("arguments")
            if sql not in exact_scores:
                exact_scores[sql] = score_sql(record["db_path"], sql, record["gold_sql"], timeout=30)
            actual = exact_scores[sql]
            supervised_sql_scores[str(position)] = actual
            if not actual.get("ok") or not actual.get("correct") or actual.get("infrastructure_error"):
                reasons.append("supervised_sql_full_score_failed")
        if scored.get("gold_rows") == 0:
            warnings.append("empty_gold_result_requires_semantic_review")
        if scored.get("ok") and scored.get("gold_rows"):
            from tools.sql_execute import run_sql
            preview = run_sql(record["db_path"], record["gold_sql"], timeout=30)
            if not preview.get("ok") or preview.get("infrastructure_error"):
                reasons.append("gold_preview_failed")
            elif preview.get("preview") and all(v is None for r in preview["preview"] for v in r):
                warnings.append("all_null_gold_preview_requires_semantic_review")
    if runtime is not None:
        if row and (not segments or segments[0].get("text") != runtime.prompt_for_row(row)
                    or record.get("prompt") != runtime.prompt_for_row(row)):
            reasons.append("prompt_contract_mismatch")
        from e1_alignment.validate import replay
        reasons.extend(replay(record, runtime))
        c = runtime.contract
        if len(flat_ids) > c.max_context:
            reasons.append("context_budget_exceeded")
        actions = [s for s in segments if s.get("kind") == "action"]
        if len(actions) > c.max_rounds or sum(s.get("action") != "answer" for s in actions) > c.max_tools:
            reasons.append("tool_or_round_budget_exceeded")
        for name, limit in (("schema_lookup", c.max_schema_lookups), ("sql_execute", c.max_sql_executes)):
            if sum(s.get("action") == name for s in actions) > limit:
                reasons.append(f"{name}_budget_exceeded")
        for s in segments:
            if runtime.tokenizer.decode(s.get("ids", []), skip_special_tokens=True) != s.get("text"):
                reasons.append("segment_decode_mismatch")
            if s.get("kind") == "observation" and len(s.get("ids", [])) > c.max_observation_tokens:
                reasons.append("observation_budget_exceeded")
            if s.get("kind") == "action" and len(s.get("ids", [])) > c.max_action_tokens:
                reasons.append("action_budget_exceeded")
    else:
        reasons.append("runtime_replay_not_performed")
    return {"ok": not reasons, "content_sha256": record.get("content_sha256"),
            "rule_version": rule_version, "ast_rule_version": AST_RULE_VERSION,
            "parser_version": PARSER_VERSION, "reasons": sorted(set(reasons)),
            "warnings": warnings, "ast": asts, "visible_coverage": checks,
            "full_score": scored, "supervised_sql_scores": supervised_sql_scores, "prefix_failure": prefix_failure, "replayed": runtime is not None}


def review_errors(record, review, *, rule_version=RULE_VERSION):
    errors = []
    if record.get("quality_rule_version") != rule_version:
        errors.append("record_quality_rule_mismatch")
    if not review or review.get("status") != "pass":
        return ["unreviewed_or_not_passed"]
    if review.get("trajectory_id") != record.get("trajectory_id") or review.get("content_sha256") != content_hash(record):
        errors.append("stale_or_wrong_review")
    if review.get("rule_version") != rule_version:
        errors.append("review_rule_version_mismatch")
    if not review.get("reviewer") or not review.get("reviewed_at") or review.get("method") != "individual_assistant_review":
        errors.append("missing_individual_review_provenance")
    if not isinstance(review.get("review_seconds"), (int, float)) or review["review_seconds"] <= 0:
        errors.append("missing_review_duration")
    for field in REVIEW_FIELDS:
        item = review.get(field) or {}
        if item.get("verdict") != "pass" or not str(item.get("notes") or "").strip():
            errors.append(f"review_incomplete:{field}")
    required = {str(j) for ast in (record.get("program_check") or {}).get("ast", {}).values() for j in ast.get("joins", [])}
    verified = review.get("join_verification") or []
    for key in required:
        items = [v for v in verified if v.get("join_key") == key and v.get("verdict") == "pass"
                 and v.get("semantic_basis") and v.get("independent_evidence")]
        if not items:
            errors.append("join_semantics_not_individually_verified")
    check = record.get("program_check") or {}
    if check.get("parser_version") != PARSER_VERSION or check.get("ast_rule_version") != AST_RULE_VERSION:
        errors.append("program_parser_rule_mismatch")
    if not check.get("ok") or check.get("content_sha256") != content_hash(record) or check.get("rule_version") != rule_version:
        errors.append("program_check_failed_or_stale")
    return sorted(set(errors))
