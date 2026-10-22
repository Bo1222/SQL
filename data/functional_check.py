"""Functional checks for a built JSONL: gold executability, schema, prompt length."""
import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402


def sqlite_shape(db_path: str) -> tuple[set[str], int]:
    con = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if not row[0].startswith("sqlite_")
        }
        columns = 0
        for table in tables:
            columns += len(con.execute(f'PRAGMA table_info("{table}")').fetchall())
        return tables, columns
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dev_200.jsonl")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--out", default="reports/functional_check_200.json")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    report = {
        "data": args.data,
        "n": len(rows),
        "gold_ok": 0,
        "gold_fail": [],
        "schema_mismatch": [],
        "prompt_tokens": [],
        "over_limit": 0,
        "duplicate_questions": 0,
        "db_counts": Counter(r["db_id"] for r in rows),
        "difficulty_counts": Counter(r.get("difficulty", "") for r in rows),
    }

    seen = set()
    for row in rows:
        result = execute(row["db_path"], row["gold_sql"])
        if result.ok:
            report["gold_ok"] += 1
        else:
            report["gold_fail"].append(
                {"id": row["id"], "db_id": row["db_id"], "error": result.error}
            )

        tables, columns = sqlite_shape(row["db_path"])
        if len(tables) != row["n_tables"] or columns != row["n_columns"]:
            report["schema_mismatch"].append(
                {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "expected": [row["n_tables"], row["n_columns"]],
                    "actual": [len(tables), columns],
                }
            )

        if tokenizer is not None:
            n_tokens = len(tokenizer(row["prompt"])["input_ids"])
        else:
            n_tokens = len(row["prompt"]) // 4
        report["prompt_tokens"].append(n_tokens)
        if n_tokens > args.max_prompt_tokens:
            report["over_limit"] += 1

        if row["question"] in seen:
            report["duplicate_questions"] += 1
        seen.add(row["question"])

    tokens = sorted(report["prompt_tokens"])
    report["prompt_token_stats"] = {
        "min": tokens[0] if tokens else 0,
        "p50": tokens[len(tokens) // 2] if tokens else 0,
        "p95": tokens[int(len(tokens) * 0.95) - 1] if tokens else 0,
        "max": tokens[-1] if tokens else 0,
    }
    report["db_counts"] = dict(report["db_counts"])
    report["difficulty_counts"] = dict(report["difficulty_counts"])
    report["gold_ok_rate"] = round(report["gold_ok"] / len(rows), 4) if rows else 0.0
    report["schema_mismatch_rate"] = round(len(report["schema_mismatch"]) / len(rows), 4) if rows else 0.0
    report["over_limit_rate"] = round(report["over_limit"] / len(rows), 4) if rows else 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "n",
                    "gold_ok_rate",
                    "schema_mismatch_rate",
                    "prompt_token_stats",
                    "over_limit_rate",
                    "duplicate_questions",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
