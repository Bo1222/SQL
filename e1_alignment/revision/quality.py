import copy
from e1_alignment.common import canonical_hash
from e1_alignment.rebuild.quality import (content_hash, program_check as legacy_check,
                                        review_errors as legacy_review, failure_prefix_check)
from e1_alignment.revision import RULE_VERSION
from e1_alignment.revision.policy import policy_hash
from e1_alignment.revision.provenance import verify_history


def prepare(record, policy):
    record = copy.deepcopy(record)
    record.pop("semantic_review", None)
    record["quality_rule_version"] = RULE_VERSION
    record["data_policy_sha256"] = policy_hash(policy)
    record["content_sha256"] = content_hash(record)
    return record


def error_class(result, score):
    if result.get("status") == "ok":
        return "semantic"
    text = str(result).lower()
    if "column" in text or "ambiguous" in text or "alias" in text:
        return "column_alias"
    if "table" in text or "join" in text:
        return "table_join"
    if "syntax" in text or "incomplete input" in text or "unrecognized token" in text:
        return "syntax"
    return "other"


def verified_repair(record, score_fn):
    """Derive repair status from real failure + subsequent scored correct execute."""
    actions = [s for s in record["segments"] if s.get("kind") == "action"]
    def sql_result(action):
        index = action.get("tool_event_index")
        events = record.get("tool_events", [])
        if action.get("invalid") or type(index) is not int or not 0 <= index < len(events):
            return None
        event = events[index]
        if event.get("action") != "sql_execute" or event.get("arguments") != action.get("arguments"):
            return None
        return event.get("result")

    gold = score_fn(record["db_path"], record["gold_sql"], record["gold_sql"])
    if not gold.get("ok") or not gold.get("correct") or gold.get("infrastructure_error"):
        return {"verified": False}
    for index, action in enumerate(actions):
        if action.get("action") != "sql_execute":
            continue
        result = sql_result(action)
        if not isinstance(result, dict):
            # Malformed generated actions have no real execution receipt. Their
            # sequence check rejects the record; they must not crash the audit
            # or become a semantic failure merely through offline scoring.
            continue
        failed = failure_prefix_check({**record, "prefix_action_segments": index + 1}, score_fn)
        if not failed["ok"]:
            continue
        if result.get("failure_origin") == "infrastructure" or result.get("error_type") in ("budget", "timeout", "scorer_failure"):
            continue
        for later in actions[index + 1:]:
            if later.get("action") != "sql_execute" or not any(later.get("loss_mask", [])):
                continue
            later_result = sql_result(later)
            if not isinstance(later_result, dict) or later_result.get("status") != "ok":
                continue
            score = score_fn(record["db_path"], later["arguments"], record["gold_sql"])
            if score.get("ok") and score.get("correct") and not score.get("infrastructure_error"):
                return {"verified": True, "failure_action_count": index + 1,
                        "error_class": error_class(result, failed.get("score")), "failure": failed}
    return {"verified": False}


def check(record, policy, train, monitor, fingerprint, runtime, forbidden):
    from e1_alignment.revision.scoring import score_sql
    reasons = []
    try:
        verify_history(record)
    except (KeyError, ValueError, OSError, IndexError) as exc:
        reasons.append("source_not_verified:" + str(exc))
    if record.get("data_policy_sha256") != policy_hash(policy):
        reasons.append("data_policy_mismatch")
    gold = record.get("construction_source") == "gold_assisted"
    if record.get("construction_source") not in ("gold_assisted", "model_generated"):
        reasons.append("invalid_construction_source")
    if gold != (record.get("gold_used_for_construction") is True):
        reasons.append("construction_gold_flag_mismatch")
    if gold and record.get("autonomous_repair") is not False:
        reasons.append("gold_assisted_marked_autonomous")
    if record.get("revision", 0) > 1:
        reasons.append("one_revision_limit")
    if record.get("unverified_independent_failure_origin"):
        reasons.append("independent_failure_origin_not_proven")
    mode = record.get("start_mode")
    if mode not in ("question_start", "failure_continuation"):
        reasons.append("invalid_start_mode")
    if mode == "question_start" and record.get("prefix_action_segments", 0):
        reasons.append("question_start_with_masked_prefix")
    answers = [s for s in record.get("segments", []) if s.get("action") == "answer"]
    if len(answers) != 1 or answers[0].get("arguments") != record.get("final_sql"):
        reasons.append("original_answer_sql_mismatch")
    prefixes = {}
    if mode == "failure_continuation":
        n = record.get("prefix_action_segments", 0)
        state = (record.get("failure_state") or {}).get("state_id")
        prefixes[state] = [{"segments": record["segments"][1:1 + 2 * n]}]
    result = legacy_check(record, train, monitor, fingerprint, runtime=runtime,
                          forbidden_dbs=forbidden, prefix_index=prefixes,
                          rule_version=RULE_VERSION, failure_continuation=mode == "failure_continuation", score_function=score_sql)
    repair = verified_repair(record, score_sql)
    if record.get("has_verified_failure_repair") != repair["verified"]:
        reasons.append("repair_flag_mismatch")
    if repair["verified"] and record.get("repair_evidence") != repair:
        reasons.append("repair_evidence_mismatch")
    if mode == "failure_continuation" and not repair["verified"]:
        reasons.append("continuation_without_verified_repair")
    result["reasons"] = sorted(set(result["reasons"] + reasons))
    result["ok"] = not result["reasons"]
    result["data_policy_sha256"] = policy_hash(policy)
    return result


def review_errors(record, review, policy):
    errors = legacy_review(record, review, rule_version=RULE_VERSION)
    if not review:
        return errors
    if review.get("data_policy_sha256") != policy_hash(policy):
        errors.append("review_policy_mismatch")
    if review.get("program_check_sha256") != canonical_hash(record.get("program_check")):
        errors.append("review_program_check_changed")
    return errors
