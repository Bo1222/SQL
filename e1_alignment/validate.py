"""E1-05: replay-validate, dedupe and stratify the E1 alignment SFT dataset."""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from agent.protocol import parse_action
from agent.sql_episode import encode_observation
from e1_alignment.common import (DEV200, DEV200_MANIFEST, RL_TRAIN, ROOT, canonical_hash,
                                 normalize_sql, read_jsonl, safe_div, sha256_file, write_json,
                                 write_jsonl)
from e1_alignment.dataset import apply_loss_mask, dedupe_key, score_records
from e1_alignment.runtime import Runtime, RuntimeContract

DEFAULT_A_RAW = "artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl"
DEFAULT_A_PASS = "artifacts/m6/e1_alignment/e1_self_rollouts.jsonl"
DEFAULT_B_RAW = "artifacts/m6/e1_alignment/e1_repair_rollouts.raw.jsonl"
DEFAULT_B_PASS = "artifacts/m6/e1_alignment/e1_repair_rollouts.jsonl"
DEFAULT_C_RAW = "artifacts/m6/e1_alignment/e1_gold_assisted.raw.jsonl"
DEFAULT_C_PASS = "artifacts/m6/e1_alignment/e1_gold_assisted.jsonl"
DEFAULT_OUT = "artifacts/m6/e1_alignment/e1_alignment_sft.jsonl"
DEFAULT_MANIFEST = "artifacts/m6/e1_alignment/dataset_manifest.json"
DEFAULT_REPORT = "reports/m6/e1_alignment/validation_report.json"
DEFAULT_REJECT = "reports/m6/e1_alignment/rejected_records.jsonl"


def replay(record, runtime, require_exact_observation=True):
    """Replay every stored action against the same E1 runtime."""
    reasons = []
    segments = record.get("segments") or []
    if not segments or segments[0].get("kind") != "prompt":
        return ["missing_prompt"]
    contract = runtime.contract
    prompt_ids = runtime.tokenizer.encode(segments[0].get("text") or "", add_special_tokens=False)
    if prompt_ids != list(segments[0].get("ids") or []):
        reasons.append("prompt_ids_mismatch")
    if record.get("prompt_ids") and list(record["prompt_ids"]) != prompt_ids:
        reasons.append("record_prompt_ids_mismatch")
    tool_index = 0
    final_sql = None
    saw_answer = False
    for position, segment in enumerate(segments):
        kind = segment.get("kind")
        ids = list(segment.get("ids") or [])
        text = segment.get("text") or ""
        if kind == "prompt":
            continue
        if kind == "observation":
            continue
        if kind != "action":
            reasons.append(f"unknown_segment_kind:{kind}")
            continue
        if segment.get("invalid"):
            reasons.append("invalid_action_in_positive")
            continue
        if runtime.tokenizer.decode(ids, skip_special_tokens=True) != text:
            reasons.append("action_decode_mismatch")
        try:
            action = parse_action(text)
        except (ValueError, TypeError) as exc:
            reasons.append(f"action_parse_error:{str(exc)[:60]}")
            continue
        if action.name == "answer":
            final_sql = action.arguments
            saw_answer = True
            if position != len(segments) - 1:
                reasons.append("answer_not_last")
            break
        try:
            if action.name == "schema_lookup":
                payload = runtime.schema_tool(record["db_id"])(action.arguments)
            else:
                payload = runtime.sql_tool(record["db_path"])(action.arguments)
        except Exception as exc:
            reasons.append(f"replay_tool_error:{str(exc)[:80]}")
            continue
        expected = segments[position + 1] if position + 1 < len(segments) else None
        if not expected or expected.get("kind") != "observation":
            reasons.append("missing_observation")
            continue
        obs_ids, obs_text = encode_observation(runtime.tokenizer, payload,
                                               contract.max_observation_tokens,
                                               mode=contract.observation_mode)
        if require_exact_observation and (obs_text != expected.get("text") or obs_ids != list(expected.get("ids") or [])):
            reasons.append("observation_mismatch")
        stored = (record.get("tool_events") or [])
        if tool_index < len(stored):
            event = stored[tool_index]
            if event.get("action") != action.name or event.get("arguments") != action.arguments:
                reasons.append("tool_event_mismatch")
            original_result = event.get("result") or {}
            if (original_result.get("status") or "ok") != (payload.get("status") or "ok"):
                reasons.append("tool_status_mismatch")
        else:
            reasons.append("missing_tool_event")
        tool_index += 1
    if not saw_answer:
        reasons.append("missing_answer")
    if final_sql is None:
        reasons.append("missing_final_sql")
    elif normalize_sql(final_sql) != normalize_sql(record.get("final_sql") or ""):
        reasons.append("final_sql_mismatch")
    return reasons


def validate_record(record, runtime, dev_ids, rl_rows, expected_fingerprint):
    reasons = list(replay(record, runtime))
    gold = normalize_sql(record.get("gold_sql") or "")
    prompt = normalize_sql(record.get("prompt") or "")
    if gold and gold in prompt:
        reasons.append("gold_in_prompt")
    for observation in record.get("observations") or []:
        if gold and gold in normalize_sql(observation):
            reasons.append("gold_in_observation")
    for field in ("id", "db_id", "trajectory_source", "policy_checkpoint",
                  "runtime_fingerprint_sha256", "final_sql", "final_correct",
                  "loss_mask", "gold_used_for_construction"):
        if field not in record:
            reasons.append(f"missing_field:{field}")
    if record.get("id") in dev_ids:
        reasons.append("dev200_id_in_training")
    row = rl_rows.get(record.get("id"))
    if row is None:
        reasons.append("not_in_rl_train")
    else:
        if row.get("db_id") != record.get("db_id"):
            reasons.append("db_id_mismatch")
        if row.get("db_path") != record.get("db_path"):
            reasons.append("db_path_mismatch")
    if record.get("runtime_fingerprint_sha256") != expected_fingerprint:
        reasons.append("runtime_fingerprint_mismatch")
    source = record.get("trajectory_source")
    if source == "e1_gold_assisted":
        if not record.get("gold_used_for_construction"):
            reasons.append("gold_assisted_missing_flag")
        if record.get("autonomous_repair"):
            reasons.append("gold_assisted_marked_autonomous_repair")
    elif record.get("gold_used_for_construction"):
        reasons.append("non_gold_source_marked_gold_assisted")
    if not record.get("final_correct"):
        reasons.append("final_not_correct")
    ids, mask = list(record.get("input_ids") or []), list(record.get("loss_mask") or [])
    if len(ids) != len(mask):
        reasons.append("loss_mask_length_mismatch")
    if len(ids) != sum(len(segment.get("ids") or []) for segment in record.get("segments") or []):
        reasons.append("input_ids_segment_length_mismatch")
    for segment in record.get("segments") or []:
        if len(segment.get("loss_mask") or []) != len(segment.get("ids") or []):
            reasons.append("segment_mask_length_mismatch")
        if segment.get("kind") != "action" and any(segment.get("loss_mask") or []):
            reasons.append("non_action_supervised")
    if not any(mask):
        reasons.append("no_supervised_tokens")
    if record.get("trajectory_source") == "e1_repair_continuation":
        prefix = int(record.get("prefix_action_segments") or 0)
        actions = [segment for segment in record["segments"] if segment.get("kind") == "action"]
        if not all(not any(action.get("loss_mask") or []) for action in actions[:prefix]):
            reasons.append("failed_prefix_supervised")
    return sorted(set(reasons))


def cross_dedupe_key(record):
    actions = [(segment.get("action"), normalize_sql(segment.get("arguments") or ""))
               for segment in record.get("segments") or [] if segment.get("kind") == "action"]
    return json.dumps({"question_id": record.get("id"), "actions": actions,
                       "final_sql": normalize_sql(record.get("final_sql") or "")},
                      ensure_ascii=False, sort_keys=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-raw", default=DEFAULT_A_RAW)
    p.add_argument("--a-pass", default=DEFAULT_A_PASS)
    p.add_argument("--b-raw", default=DEFAULT_B_RAW)
    p.add_argument("--b-pass", default=DEFAULT_B_PASS)
    p.add_argument("--c-raw", default=DEFAULT_C_RAW)
    p.add_argument("--c-pass", default=DEFAULT_C_PASS)
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--rl-data", default=str(RL_TRAIN))
    p.add_argument("--fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--reject-out", default=DEFAULT_REJECT)
    a = p.parse_args()

    runtime = Runtime(RuntimeContract(model_path=a.model))
    dev_manifest = json.loads(DEV200_MANIFEST.read_text(encoding="utf-8"))
    dev_ids = set(dev_manifest.get("ids") or [])
    rl_rows = {row["id"]: row for row in read_jsonl(a.rl_data)}
    expected_fingerprint = json.loads(Path(a.fingerprint).read_text(encoding="utf-8"))["fingerprint_sha256"]
    sources = [("e1_self_rollout", a.a_raw, a.a_pass),
               ("e1_repair_continuation", a.b_raw, a.b_pass),
               ("e1_gold_assisted", a.c_raw, a.c_pass)]
    accepted, rejected = [], []
    funnel = {}
    started = time.monotonic()
    by_source = {}
    for source, raw_path, pass_path in sources:
        raw = read_jsonl(raw_path)
        positive = read_jsonl(pass_path)
        funnel[source] = {"raw_candidates": len(raw), "positive_candidates": len(positive)}
        kept = 0
        for record in positive:
            reasons = validate_record(record, runtime, dev_ids, rl_rows, expected_fingerprint)
            if reasons:
                rejected.append({"id": record.get("id"), "source": source, "reasons": reasons})
            else:
                accepted.append(record)
                kept += 1
        funnel[source]["replay_passed"] = kept
        funnel[source]["replay_failed"] = len(positive) - kept
        by_source[source] = kept
        print(json.dumps({"source": source, "raw": len(raw), "positive": len(positive),
                          "replay_passed": kept}), flush=True)

    # Cap each source/question at four positives so one solved question cannot
    # dominate the mixture (A already samples at most four candidates).
    per_source_question = Counter()
    capped = []
    for record in accepted:
        key = (record.get("trajectory_source"), record.get("id"))
        if per_source_question[key] >= 4:
            continue
        per_source_question[key] += 1
        capped.append(record)
    cap_removed = len(accepted) - len(capped)

    # Dedupe with source priority A > B > C (preserved by input order).
    deduped, seen, duplicates = [], set(), 0
    for record in capped:
        key = cross_dedupe_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(record)

    # Re-score once against the gold execution oracle and drop scorer failures.
    score_records(deduped)
    scored_ok = [record for record in deduped if record.get("final_correct")
                 and not record.get("final_infrastructure_error")]
    score_failed = len(deduped) - len(scored_ok)

    # Keep Gold-assisted C at or below 25% of the final mixture (plan target is
    # 20%); downsample deterministically by DB rather than duplicating A/B.
    source_counts = Counter(record["trajectory_source"] for record in scored_ok)
    c_count = source_counts.get("e1_gold_assisted", 0)
    ab_count = len(scored_ok) - c_count
    c_cap = max(0, ab_count // 3) if ab_count else c_count
    c_downsampled = 0
    if c_count > c_cap:
        c_records = [record for record in scored_ok
                     if record["trajectory_source"] == "e1_gold_assisted"]
        others = [record for record in scored_ok
                  if record["trajectory_source"] != "e1_gold_assisted"]
        buckets = defaultdict(list)
        for record in c_records:
            buckets[record["db_id"]].append(record)
        for db in buckets:
            buckets[db].sort(key=lambda record: record.get("dedupe_key") or record.get("id"))
        chosen, dbs = [], sorted(buckets)
        while len(chosen) < c_cap:
            progressed = False
            for db in dbs:
                if buckets[db] and len(chosen) < c_cap:
                    chosen.append(buckets[db].pop(0))
                    progressed = True
            if not progressed:
                break
        c_downsampled = c_count - len(chosen)
        scored_ok = others + chosen

    for record in scored_ok:
        record["question_id"] = record.get("id")
        record["runtime_version"] = "E1"
        apply_loss_mask(record, prefix_action_segments=int(record.get("prefix_action_segments") or 0))
        record["dedupe_key"] = dedupe_key(record)

    write_jsonl(a.out, scored_ok)
    write_jsonl(a.reject_out, rejected)

    def shape(record):
        lookups = sum(1 for event in record.get("tool_events") or []
                      if event.get("action") == "schema_lookup")
        executes = sum(1 for event in record.get("tool_events") or []
                       if event.get("action") == "sql_execute")
        if lookups and executes:
            return "lookup_and_execute"
        if lookups:
            return "lookup_only"
        if executes:
            return "execute_only"
        return "zero_tool"

    by_source_final = Counter(record["trajectory_source"] for record in scored_ok)
    by_success = Counter(record.get("success_type") for record in scored_ok)
    by_db = Counter(record["db_id"] for record in scored_ok)
    by_shape = Counter(shape(record) for record in scored_ok)
    by_lookups = Counter(record.get("schema_lookups", 0) for record in scored_ok)
    by_failure = Counter((record.get("failure_state") or {}).get("error_type")
                         for record in scored_ok if record["trajectory_source"] == "e1_repair_continuation")
    total = len(scored_ok)
    token_stats = {
        "input_tokens": sum(len(record["input_ids"]) for record in scored_ok),
        "supervised_tokens": sum(record["supervised_tokens"] for record in scored_ok),
        "avg_input_tokens": safe_div(sum(len(record["input_ids"]) for record in scored_ok), total),
        "avg_supervised_tokens": safe_div(sum(record["supervised_tokens"] for record in scored_ok), total),
        "max_input_tokens": max((len(record["input_ids"]) for record in scored_ok), default=0),
        "supervised_token_ratio": safe_div(sum(record["supervised_tokens"] for record in scored_ok),
                                           sum(len(record["input_ids"]) for record in scored_ok)),
    }
    manifest = {
        "format": "sqlrl-e1-alignment-sft-v1",
        "n": total,
        "by_source": dict(by_source_final),
        "source_ratio": {key: round(value / max(total, 1), 4) for key, value in by_source_final.items()},
        "by_success_type": dict(by_success), "by_db": dict(by_db),
        "by_tool_shape": dict(by_shape), "by_schema_lookups": {str(k): v for k, v in by_lookups.items()},
        "by_failure_error_type": {str(k): v for k, v in by_failure.items()},
        "funnel": funnel,
        "dedupe": {"input": len(accepted), "per_source_question_cap_removed": cap_removed,
                   "after_cap": len(capped), "duplicates_removed": duplicates,
                   "kept": len(deduped)},
        "gold_assisted_downsample": {"original": c_count, "cap": c_cap,
                                     "removed": c_downsampled,
                                     "kept": source_counts.get("e1_gold_assisted", 0) - c_downsampled},
        "scorer": {"rechecked": len(deduped), "failed": score_failed, "kept": total},
        "token_stats": token_stats,
        "dataset_sha256": sha256_file(ROOT / a.out),
        "runtime_fingerprint_sha256": next((record.get("runtime_fingerprint_sha256")
                                            for record in scored_ok), None),
        "dev_manifest_sha256": sha256_file(DEV200_MANIFEST),
        "dev_ids_sha256": dev_manifest.get("sha256_sorted_ids"),
        "split_audit_sha256": sha256_file(ROOT / "artifacts/manifests/split_audit.json"),
        "validation": {"replay_passed": len(accepted), "replay_failed": len(rejected),
                       "per_source_question_cap_removed": cap_removed,
                       "duplicates": duplicates, "scorer_failed": score_failed,
                       "gold_assisted_downsampled": c_downsampled, "final": total},
        "gold_assisted_ratio_note": "Gold-assisted trajectories are marked and counted separately; "
                                    "they are not autonomous success or autonomous repair.",
    }
    write_json(a.manifest, manifest)
    report = {
        "stage": "E1-05", "elapsed_s": round(time.monotonic() - started, 1),
        "funnel": funnel,
        "rejections": Counter(reason for row in rejected for reason in row["reasons"]),
        "manifest": str(a.manifest), "dataset": str(a.out),
        "records": total,
    }
    write_json(a.report, report)
    print(json.dumps({"done": True, "records": total, "by_source": dict(by_source_final),
                      "by_success": dict(by_success), "manifest": a.manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
