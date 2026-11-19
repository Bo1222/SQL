"""Build the P1/P2 task file from the 20-question stress pilot selection.

The 20 questions are the same ones used by the M6-08m root-cause pilot:
5 supervised-empty + 5 SQL-table-mismatch + 10 random.  They are SFT/train
questions only; dev200/val590/dev1534 are never included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout20", default="reports/m6/e1_contract_rebuild/rollout20/rollout20.jsonl")
    parser.add_argument("--raw", default="data/processed/bird_agent/full_sft_20261026/full_tool_unique.jsonl")
    parser.add_argument("--out", default="artifacts/m6/e1_student_only/tasks20.jsonl")
    args = parser.parse_args()

    raw_by_id = {row.get("id"): row for row in read_jsonl(args.raw)}
    tasks = []
    for row in read_jsonl(args.rollout20):
        question_id = row["question_id"]
        raw = raw_by_id.get(question_id)
        if raw is None:
            raise SystemExit(f"missing raw row for {question_id}")
        if raw.get("split") != "sft":
            raise SystemExit(f"non-SFT split for {question_id}: {raw.get('split')}")
        tasks.append({
            "id": question_id,
            "question_id": question_id,
            "source_trajectory_id": row.get("trajectory_id"),
            "group": row.get("group"),
            "split": "sft",
            "db_id": raw["db_id"],
            "db_path": raw["db_path"],
            "question": raw["question"],
            "evidence": raw.get("evidence", ""),
            "gold_sql": raw["gold_sql"],
        })

    # Shuffle deterministically so card shards are not group-ordered.
    import random
    random.Random(0).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["task_index"] = index

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "n": len(tasks),
                      "groups": {g: sum(t["group"] == g for t in tasks)
                                 for g in sorted({t["group"] for t in tasks})}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
