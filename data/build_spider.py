"""Build Spider 1.0 splits (test/val/sft/rl) with cleaning and overlap checks."""
import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schema import build_schema  # noqa: E402
from env.sqlite_exec import execute  # noqa: E402

PROMPT = """You are a SQLite expert. Given the database schema and a question, write one SQL query that answers it.
Return only the final SQL inside <answer> and </answer>, with no other text inside the tags.

### Database schema
{schema}

### Question
{question}

### Answer
"""


def normalize_question(text: str) -> str:
    return " ".join((text or "").lower().split())


def is_single_statement(sql: str) -> bool:
    """True if sql contains at most one statement; semicolons in strings are ignored."""
    text = (sql or "").strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    count, quote_char = 0, None
    for char in text:
        if quote_char:
            if char == quote_char:
                quote_char = None
        elif char in ("'", '"'):
            quote_char = char
        elif char == ";":
            count += 1
    return count == 0


def sqlite_shape(db_path: str) -> tuple[int, int]:
    con = sqlite3.connect("file:" + str(Path(db_path).resolve()) + "?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if not row[0].startswith("sqlite_")
        ]
        columns = sum(len(con.execute(f'PRAGMA table_info("{t}")').fetchall()) for t in tables)
        return len(tables), columns
    finally:
        con.close()


def build_records(raw, tables_map, db_dir, tokenizer, max_prompt_tokens, source):
    kept, dropped, mismatch = [], Counter(), set()
    seen = set()
    for idx, item in enumerate(raw):
        db_id = item.get("db_id", "")
        question = (item.get("question") or "").strip()
        gold = (item.get("query") or "").strip()
        if not question or not gold:
            dropped["empty"] += 1
            continue
        key = (db_id, normalize_question(question))
        if key in seen:
            dropped["duplicate_question"] += 1
            continue
        seen.add(key)
        if not is_single_statement(gold):
            dropped["gold_multi_statement"] += 1
            continue
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            dropped["missing_db"] += 1
            continue
        result = execute(str(db_path), gold, timeout_s=10.0)
        if not result.ok:
            dropped["gold_exec_fail"] += 1
            continue

        entry = tables_map.get(db_id)
        if entry is None:
            dropped["missing_schema"] += 1
            continue
        schema = build_schema(entry)
        prompt = PROMPT.format(schema=schema, question=question)
        n_tokens = len(tokenizer(prompt)["input_ids"]) if tokenizer is not None else None
        if n_tokens is not None and n_tokens > max_prompt_tokens:
            dropped["prompt_too_long"] += 1
            continue

        expected_tables = len(entry["table_names_original"])
        expected_columns = len([c for c in entry["column_names_original"] if c[0] >= 0])
        actual_tables, actual_columns = sqlite_shape(str(db_path))
        has_mismatch = expected_tables != actual_tables or expected_columns != actual_columns
        if has_mismatch:
            mismatch.add(db_id)

        kept.append(
            {
                "id": f"spider_{source}_{idx:06d}",
                "db_id": db_id,
                "db_path": str(db_path),
                "question": question,
                "gold_sql": gold,
                "prompt": prompt,
                "source": source,
                "n_tables": expected_tables,
                "n_columns": expected_columns,
                "schema_mismatch": has_mismatch,
                "prompt_tokens": n_tokens,
            }
        )
    return kept, dropped, mismatch


def split_by_question(items, sft_ratio, seed):
    by_db = defaultdict(list)
    for item in items:
        by_db[item["db_id"]].append(item)
    rng = random.Random(seed)
    sft, rl = [], []
    for db_id in sorted(by_db):
        group = by_db[db_id]
        rng.shuffle(group)
        cut = max(1, int(len(group) * sft_ratio))
        sft.extend(group[:cut])
        rl.extend(group[cut:])
    return sft, rl


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_stats(rows):
    values = sorted(row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None)
    if not values:
        return None
    return {
        "min": values[0],
        "p50": values[len(values) // 2],
        "p95": values[int(len(values) * 0.95) - 1],
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/spider_extract/spider_data")
    parser.add_argument("--out", default="data/processed/spider")
    parser.add_argument("--val-dbs", type=int, default=20)
    parser.add_argument("--sft-ratio", type=float, default=0.8)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.root)
    db_dir = root / "database"
    tables = json.loads((root / "tables.json").read_text(encoding="utf-8"))
    tables_map = {entry["db_id"]: entry for entry in tables}
    train_raw = json.loads((root / "train_spider.json").read_text(encoding="utf-8"))
    dev_raw = json.loads((root / "dev.json").read_text(encoding="utf-8"))

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    train, train_dropped, train_mismatch = build_records(
        train_raw, tables_map, db_dir, tokenizer, args.max_prompt_tokens, "train"
    )
    test, test_dropped, test_mismatch = build_records(
        dev_raw, tables_map, db_dir, tokenizer, args.max_prompt_tokens, "dev"
    )

    rng = random.Random(args.seed)
    all_dbs = sorted({row["db_id"] for row in train})
    val_dbs = set(rng.sample(all_dbs, min(args.val_dbs, len(all_dbs))))
    val = [row for row in train if row["db_id"] in val_dbs]
    pool = [row for row in train if row["db_id"] not in val_dbs]
    sft, rl = split_by_question(pool, args.sft_ratio, args.seed)

    for split_name, rows in (("test", test), ("val", val), ("sft", sft), ("rl", rl)):
        for row in rows:
            row["split"] = split_name

    sft_keys = {(row["db_id"], normalize_question(row["question"])) for row in sft}
    rl_keys = {(row["db_id"], normalize_question(row["question"])) for row in rl}
    train_dbs = {row["db_id"] for row in sft} | {row["db_id"] for row in rl}
    test_dbs = {row["db_id"] for row in test}

    report = {
        "root": str(root),
        "elapsed_s": round(time.time() - started, 1),
        "counts": {
            "sft": len(sft),
            "rl": len(rl),
            "val": len(val),
            "test": len(test),
        },
        "db_counts": {
            "sft": len({row["db_id"] for row in sft}),
            "rl": len({row["db_id"] for row in rl}),
            "val": len(val_dbs),
            "test": len(test_dbs),
        },
        "dropped": {"train": dict(train_dropped), "test": dict(test_dropped)},
        "overlaps": {
            "val_vs_train_dbs": len(val_dbs & train_dbs),
            "test_vs_train_dbs": len(test_dbs & train_dbs),
            "sft_vs_rl_questions": len(sft_keys & rl_keys),
        },
        "schema_mismatch_dbs": sorted(train_mismatch | test_mismatch),
        "prompt_tokens": {
            "sft": token_stats(sft),
            "rl": token_stats(rl),
            "val": token_stats(val),
            "test": token_stats(test),
        },
        "val_dbs": sorted(val_dbs),
    }

    out = Path(args.out)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "rl.jsonl", rl)
    (out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
