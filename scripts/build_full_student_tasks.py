"""Build the full M6-08m student-only task file from the SFT/train split.

Source: ``full_tool.jsonl`` (4,471 SFT records, each with question, evidence,
gold SQL and db_path).  The old provenance is kept only as ``source_origin``
for audit; it does not enter the new trajectory provenance.

The builder verifies:
* split == "sft";
* unique ids;
* gold SQL and database path present;
* zero overlap with dev200/val590/dev1534 by question id and by database;
* optional exact duplicate removal by (db_id, question, gold_sql).

The order is shuffled with a fixed seed so each card sees a mix of databases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/processed/bird_agent/full_sft_20261026/full_tool.jsonl")
    parser.add_argument("--dev200", default="artifacts/datasets/dev200.jsonl")
    parser.add_argument("--val", default="data/processed/bird_agent/val.jsonl")
    parser.add_argument("--test", default="data/processed/bird_agent/test.jsonl")
    parser.add_argument("--out", default="artifacts/m6/e1_student_only/tasks4190.jsonl")
    parser.add_argument("--manifest", default="artifacts/m6/e1_student_only/tasks4190_manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-rollouts", action="append", default=[],
                        help="rollout JSONL whose questions are already generated and should be skipped")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    manifest_path = Path(args.manifest)
    if (out.exists() or manifest_path.exists()) and not args.overwrite:
        raise SystemExit(f"{out} or {manifest_path} exists; pass --overwrite to replace")
    rows = list(read_jsonl(args.raw))
    excluded_ids = set()
    for path in args.exclude_rollouts:
        excluded_ids.update(row.get("question_id") for row in read_jsonl(path)
                            if row.get("question_id"))
    excluded_canonical = set()
    if excluded_ids:
        for row in rows:
            if row.get("id") in excluded_ids:
                excluded_canonical.add((row["db_id"], row["question"].strip(),
                                        row["gold_sql"].strip()))
    eval_ids, eval_dbs = set(), set()
    for path in (args.dev200, args.val, args.test):
        for row in read_jsonl(path):
            if row.get("id"):
                eval_ids.add(row["id"])
            if row.get("db_id"):
                eval_dbs.add(row["db_id"])
    tasks, errors = [], []
    seen = set()
    seen_canonical = {}
    duplicates = []
    for row in rows:
        question_id = row.get("id")
        if not question_id or question_id in seen:
            errors.append({"id": question_id, "reason": "missing_or_duplicate_id"})
            continue
        seen.add(question_id)
        if row.get("split") != "sft":
            errors.append({"id": question_id, "reason": f"split={row.get('split')}"})
            continue
        if question_id in eval_ids:
            errors.append({"id": question_id, "reason": "eval_question_overlap"})
            continue
        if row.get("db_id") in eval_dbs:
            errors.append({"id": question_id, "reason": f"eval_db_overlap:{row.get('db_id')}"})
            continue
        if not row.get("gold_sql") or not row.get("question") or not row.get("db_path"):
            errors.append({"id": question_id, "reason": "missing_question_or_gold_or_db"})
            continue
        if not Path(row["db_path"]).is_file():
            errors.append({"id": question_id, "reason": "missing_database"})
            continue
        canonical = (row["db_id"], row["question"].strip(), row["gold_sql"].strip())
        if canonical in excluded_canonical:
            continue
        if args.dedupe and canonical in seen_canonical:
            duplicates.append({"id": question_id, "duplicate_of": seen_canonical[canonical]})
            continue
        seen_canonical.setdefault(canonical, question_id)
        tasks.append({
            "id": question_id,
            "question_id": question_id,
            "split": "sft",
            "db_id": row["db_id"],
            "db_path": row["db_path"],
            "question": row["question"],
            "evidence": row.get("evidence", ""),
            "gold_sql": row["gold_sql"],
            "source_origin": (row.get("provenance") or {}).get("origin"),
        })

    random.Random(args.seed).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["task_index"] = index

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {
        "format": "m6-08m-student-tasks-v1",
        "raw": args.raw,
        "raw_sha256": sha256_file(args.raw),
        "dev200": args.dev200,
        "dev200_sha256": sha256_file(args.dev200),
        "val": args.val,
        "val_sha256": sha256_file(args.val),
        "test": args.test,
        "test_sha256": sha256_file(args.test),
        "out": str(out),
        "out_sha256": sha256_file(out),
        "n": len(tasks),
        "seed": args.seed,
        "dedupe": args.dedupe,
        "exclude_rollouts": args.exclude_rollouts,
        "excluded_existing_questions": len(excluded_canonical),
        "duplicates_removed": len(duplicates),
        "duplicate_examples": duplicates[:20],
        "dbs": len({task["db_id"] for task in tasks}),
        "source_origin": dict(Counter(task["source_origin"] for task in tasks)),
        "eval_overlap": {
            "ids": len({task["question_id"] for task in tasks} & eval_ids),
            "dbs": len({task["db_id"] for task in tasks} & eval_dbs),
        },
        "errors": errors,
        "error_count": len(errors),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("out", "out_sha256", "n", "dbs", "source_origin", "error_count")},
                     ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
