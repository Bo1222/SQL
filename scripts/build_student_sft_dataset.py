"""Convert student-only rollout records into mask-correct SFT records.

This is the missing bridge between ``run_student_rollouts.py`` and
``e1_alignment.student_sft_train``.  A rollout is admitted only when all of
the following hold:

* final EX is true under Gold scoring;
* provenance is ``raw_success`` or ``self_repair_success`` and trainable;
* no invalid action, no infrastructure/generation error, no score error;
* finish_reason is ``answer_submitted``;
* the task is SFT/train and does not overlap dev200/val590/dev1534.

Action supervision (the key mask rule):

* ``answer``: 1 when the trajectory is a final EX success;
* ``sql_execute``: 1 only when the real tool returned status ok;
* ``schema_lookup``: 1 only when retrieval returned at least one table;
* budget-blocked, invalid and failed actions: 0;
* prompt/observation: 0.

Gold SQL is never written into ``input_ids``; it is read only for the
prompt/observation leakage check and the final-answer consistency check.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e1_alignment.provenance import (
    TRAINABLE_SOURCES,
    read_jsonl,
    validate_provenance,
)

SQL_WS = re.compile(r"\s+")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_sql(sql):
    return SQL_WS.sub(" ", (sql or "").strip()).lower().rstrip(";").strip()


def norm_question(question):
    return SQL_WS.sub(" ", (question or "").strip()).lower()


def load_tasks(path):
    return {row["question_id"]: row for row in read_jsonl(path)}


def load_ids(path):
    return {row["id"] for row in read_jsonl(path)}


def lookup_tables(result):
    metadata = result.get("retrieval_metadata") or {}
    tables = metadata.get("matched_tables")
    if tables is None:
        tables = result.get("tables")
    return tables or []


def action_supervision(segment, tool_events):
    if segment.get("kind") != "action" or segment.get("invalid"):
        return False
    action = segment.get("action")
    if action == "answer":
        return True
    if action not in ("schema_lookup", "sql_execute"):
        return False
    event_index = segment.get("tool_event_index")
    if event_index is None or event_index >= len(tool_events):
        return False  # budget-blocked or infrastructure failure
    result = tool_events[event_index].get("result") or {}
    if result.get("status") != "ok":
        return False
    if action == "schema_lookup":
        return bool(lookup_tables(result))
    return True


def classify(rollout, task, eval_ids, eval_dbs, allowed_fingerprints=None):
    if rollout.get("exception"):
        return "exception"
    score = rollout.get("score") or {}
    if rollout.get("score_error") or score.get("infrastructure_error"):
        return "score_error"
    if not rollout.get("final_ex"):
        return "not_final_ex"
    provenance = rollout.get("provenance") or {}
    if provenance.get("source") not in TRAINABLE_SOURCES or not provenance.get("trainable"):
        return "provenance_not_trainable"
    if allowed_fingerprints and provenance.get("runtime_fingerprint") not in allowed_fingerprints:
        return "runtime_fingerprint_mismatch"
    record = rollout.get("record") or {}
    if record.get("invalid_actions"):
        return "invalid_action"
    if record.get("infrastructure_error"):
        return "infrastructure_error"
    if record.get("generation_error"):
        return "generation_error"
    if record.get("finish_reason") != "answer_submitted":
        return "not_answer_submitted"
    if not score.get("correct"):
        return "score_not_correct"
    if not rollout.get("final_sql"):
        return "missing_final_sql"
    if task is None:
        return "missing_task"
    if task.get("split") != "sft":
        return "split_not_sft"
    if task["question_id"] in eval_ids:
        return "eval_question_overlap"
    if task["db_id"] in eval_dbs:
        return "eval_db_overlap"
    return None


def build_record(rollout, task, max_context):
    provenance = copy.deepcopy(rollout["provenance"])
    record = rollout["record"]
    tool_events = record.get("tool_events") or []
    segments = copy.deepcopy(record.get("segments") or [])
    if not segments or segments[0].get("kind") != "prompt":
        return None, "missing_prompt"
    prompt_ids = record.get("prompt_ids")
    if prompt_ids is not None and list(segments[0].get("ids") or []) != list(prompt_ids):
        return None, "prompt_ids_mismatch"
    if not any(seg.get("kind") == "action" and seg.get("action") == "answer"
               for seg in segments):
        return None, "missing_answer"

    final_sql = (rollout.get("final_sql") or "").strip()
    if (record.get("final_sql") or "").strip() != final_sql:
        return None, "record_final_sql_mismatch"
    answer_actions = [seg for seg in segments
                      if seg.get("kind") == "action" and seg.get("action") == "answer"]
    if not answer_actions or (answer_actions[-1].get("arguments") or "").strip() != final_sql:
        return None, "answer_sql_mismatch"

    gold_sql = (record.get("gold_sql") or "").strip()
    if gold_sql:
        for segment in segments:
            if segment.get("kind") in ("prompt", "observation"):
                if gold_sql in (segment.get("text") or ""):
                    return None, "gold_in_prompt_or_observation"

    input_ids, loss_mask = [], []
    for segment in segments:
        ids = list(segment.get("ids") or [])
        if not ids:
            return None, "empty_segment"
        supervised = action_supervision(segment, tool_events)
        segment["supervise"] = supervised
        segment.pop("payload", None)
        input_ids.extend(ids)
        loss_mask.extend([1 if supervised else 0] * len(ids))
    if len(input_ids) != len(loss_mask) or not input_ids:
        return None, "ids_mask_mismatch"
    if loss_mask[0] != 0:
        return None, "prompt_supervised"
    if not any(loss_mask):
        return None, "no_supervised_tokens"
    if len(input_ids) > max_context:
        return None, "over_context"

    provenance.update({
        "final_ex": True,
        "final_ex_source": "gold_scoring",
        "trainable": True,
        "source_rollout_id": rollout["rollout_id"],
        "task_index": rollout.get("task_index"),
    })
    try:
        validate_provenance(provenance, require_trainable=True)
    except Exception as exc:
        return None, f"invalid_provenance:{exc}"
    clean = {
        "id": f"sft::{rollout['question_id']}::{rollout['rollout_id']}",
        "question_id": rollout["question_id"],
        "db_id": rollout["db_id"],
        "db_path": record.get("db_path") or task.get("db_path"),
        "split": "sft_train",
        "source_rollout_id": rollout["rollout_id"],
        "final_sql": final_sql,
        "final_ex": True,
        "score": rollout.get("score"),
        "provenance": provenance,
        "supervision_policy": "student-only-final-ex-action-mask-v1",
        "has_lookup": any(segment.get("kind") == "action"
                          and segment.get("action") == "schema_lookup"
                          and segment.get("supervise")
                          for segment in segments),
        "token_count": len(input_ids),
        "supervised_tokens": sum(loss_mask),
        "masked_tokens": len(input_ids) - sum(loss_mask),
        "trajectory": {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "segments": segments,
            "tool_events": tool_events,
            "final_sql": final_sql,
            "final_ex": True,
            "stop_reason": record.get("finish_reason"),
            "prompt": record.get("prompt"),
            "prompt_ids": record.get("prompt_ids"),
        },
    }
    return clean, None


def record_actions(rollout):
    return [seg.get("action") for seg in (rollout.get("record") or {}).get("segments") or []
            if seg.get("kind") == "action"]


def has_lookup(rollout):
    return "schema_lookup" in record_actions(rollout)


def rank_key(rollout):
    record = rollout.get("record") or {}
    actions = [seg for seg in record.get("segments") or [] if seg.get("kind") == "action"]
    token_count = sum(len(seg.get("ids") or []) for seg in record.get("segments") or [])
    errors = sum(int(rollout.get(key) or 0) for key in ("lookup_empty", "lookup_error", "sql_error"))
    return (errors, len(actions), token_count,
            int(rollout.get("sample_index") or 0), rollout["rollout_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="append", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--max-per-question", type=int, default=2)
    parser.add_argument("--lookup-policy",
                        choices=("keep", "prefer", "require", "balanced"),
                        default="prefer",
                        help="how to treat successful trajectories without schema_lookup")
    parser.add_argument("--dev200", default="artifacts/datasets/dev200.jsonl")
    parser.add_argument("--val", default="data/processed/bird_agent/val.jsonl")
    parser.add_argument("--test", default="data/processed/bird_agent/test.jsonl")
    parser.add_argument("--require-runtime-fingerprint", action="append", default=[],
                        help="only keep rollouts with this runtime fingerprint; repeatable")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    allowed_fingerprints = set(args.require_runtime_fingerprint) or None

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    tasks = load_tasks(args.task_file)
    eval_sets = {"dev200": load_ids(args.dev200), "val": load_ids(args.val),
                 "test": load_ids(args.test)}
    eval_ids = set().union(*eval_sets.values())
    eval_dbs = set()
    for path in (args.dev200, args.val, args.test):
        eval_dbs.update(row["db_id"] for row in read_jsonl(path))

    rollouts = []
    for path in args.rollout:
        rollouts.extend(read_jsonl(path))
    exclusions = Counter()
    candidates = defaultdict(list)
    for rollout in rollouts:
        task = tasks.get(rollout.get("question_id"))
        reason = classify(rollout, task, eval_ids, eval_dbs, allowed_fingerprints)
        if reason:
            exclusions[reason] += 1
            continue
        clean, reason = build_record(rollout, task, args.max_context)
        if reason:
            exclusions[reason] += 1
            continue
        # The source SFT file contains duplicate question/gold/db triples under
        # different ids; group by the canonical triple so training data is not
        # duplicated even if the rollout task file is not deduped.
        canonical = (task["db_id"], norm_question(task["question"]),
                     norm_sql(task["gold_sql"]))
        candidates[canonical].append((rank_key(rollout), clean))

    kept = []
    policy_dropped = duplicate_sql = cap_dropped = 0
    for canonical, items in candidates.items():
        items.sort(key=lambda pair: pair[0])
        if args.lookup_policy == "require":
            filtered = [pair for pair in items if pair[1]["has_lookup"]]
            policy_dropped += len(items) - len(filtered)
            items = filtered
        elif args.lookup_policy == "prefer":
            lookup_items = [pair for pair in items if pair[1]["has_lookup"]]
            if lookup_items:
                policy_dropped += len(items) - len(lookup_items)
                items = lookup_items
        elif args.lookup_policy == "balanced":
            lookup_items = [pair for pair in items if pair[1]["has_lookup"]]
            no_lookup_items = [pair for pair in items if not pair[1]["has_lookup"]]
            if lookup_items and no_lookup_items:
                # Keep one of each mode first, then fill up to max_per_question.
                chosen_ids = {id(lookup_items[0]), id(no_lookup_items[0])}
                items = [lookup_items[0], no_lookup_items[0]] + [
                    pair for pair in items if id(pair) not in chosen_ids
                ]
        selected, seen_sql = [], set()
        for _, clean in items:
            key = norm_sql(clean["final_sql"])
            if key in seen_sql:
                duplicate_sql += 1
                continue
            if len(selected) >= args.max_per_question:
                cap_dropped += 1
                continue
            seen_sql.add(key)
            selected.append(clean)
        kept.extend(selected)

    kept.sort(key=lambda row: (row["question_id"], row["id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sources = Counter(row["provenance"]["source"] for row in kept)
    token_counts = [row["token_count"] for row in kept]
    supervised_counts = [row["supervised_tokens"] for row in kept]
    action_counts = Counter()
    supervised_action_counts = Counter()
    for row in kept:
        for segment in row["trajectory"]["segments"]:
            if segment.get("kind") != "action":
                continue
            action = segment.get("action") or "invalid"
            action_counts[action] += 1
            if segment.get("supervise"):
                supervised_action_counts[action] += 1
    per_question = Counter(row["question_id"] for row in kept)
    manifest = {
        "format": "m6-08m-student-sft-v1",
        "builder_sha256": sha256_file(__file__),
        "rollout_files": [{"path": str(path), "sha256": sha256_file(path)}
                          for path in args.rollout],
        "task_file": args.task_file,
        "task_file_sha256": sha256_file(args.task_file),
        "out": str(out),
        "out_sha256": sha256_file(out),
        "max_context": args.max_context,
        "max_per_question": args.max_per_question,
        "require_runtime_fingerprint": args.require_runtime_fingerprint,
        "n_rollouts": len(rollouts),
        "n_canonical_questions_with_candidates": len(candidates),
        "n_canonical_questions_kept": len({
            (row["db_id"], norm_question(tasks[row["question_id"]]["question"]),
             norm_sql(tasks[row["question_id"]]["gold_sql"]))
            for row in kept}),
        "n_kept": len(kept),
        "n_questions_kept": len({row["question_id"] for row in kept}),
        "n_dbs_kept": len({row["db_id"] for row in kept}),
        "source_counts": dict(sources),
        "lookup_policy": args.lookup_policy,
        "no_lookup_records": sum(1 for row in kept if not row["has_lookup"]),
        "lookup_records": sum(1 for row in kept if row["has_lookup"]),
        "runtime_fingerprints": dict(Counter(
            row["provenance"].get("runtime_fingerprint") for row in kept)),
        "model_fingerprints": dict(Counter(
            row["provenance"].get("model_fingerprint") for row in kept)),
        "retrieval_versions": dict(Counter(
            row["provenance"].get("retrieval_version") for row in kept)),
        "exclusion_counts": dict(exclusions),
        "lookup_policy_dropped": policy_dropped,
        "duplicate_sql_dropped": duplicate_sql,
        "cap_dropped": cap_dropped,
        "duplicate_or_cap_dropped": duplicate_sql + cap_dropped,
        "supervised_tokens": sum(row["supervised_tokens"] for row in kept),
        "token_stats": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "mean": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        },
        "supervised_token_stats": {
            "min": min(supervised_counts) if supervised_counts else 0,
            "max": max(supervised_counts) if supervised_counts else 0,
            "mean": round(sum(supervised_counts) / len(supervised_counts), 2)
                    if supervised_counts else 0,
        },
        "action_counts": dict(action_counts),
        "supervised_action_counts": dict(supervised_action_counts),
        "records_per_question": dict(sorted(Counter(per_question.values()).items())),
        "questions_with_multiple_records": sum(
            1 for count in per_question.values() if count > 1),
        "eval_overlap": {
            "ids": {name: len({row["question_id"] for row in kept} & ids)
                    for name, ids in eval_sets.items()},
            "dbs": len({row["db_id"] for row in kept} & eval_dbs),
        },
        "split_counts": dict(Counter(row["split"] for row in kept)),
    }
    manifest["ready_for_sft"] = bool(
        kept
        and not ({row["question_id"] for row in kept} & eval_ids)
        and not ({row["db_id"] for row in kept} & eval_dbs)
        and all(row["provenance"]["source"] in TRAINABLE_SOURCES for row in kept)
    )
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("out", "out_sha256", "n_rollouts", "n_kept", "n_questions_kept",
                       "source_counts", "exclusion_counts", "duplicate_or_cap_dropped",
                       "supervised_tokens", "eval_overlap", "ready_for_sft")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
