"""Build BIRD dev JSONL (prompt, gold SQL, db path) for evaluation."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT = """You are a SQLite expert. Given the database schema and a question, write one SQL query that answers it.
Return only the final SQL inside <answer> and </answer>, with no other text inside the tags.

### Database schema
{schema}

### Question
{question}
{evidence}### Answer
"""


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_schema(entry: dict) -> str:
    tables = entry["table_names_original"]
    columns = entry["column_names_original"]
    types = entry["column_types"]
    primary_keys = entry["primary_keys"]
    foreign_keys = entry["foreign_keys"]

    by_table = defaultdict(list)
    for idx, (table_idx, name) in enumerate(columns):
        if table_idx < 0:
            continue
        by_table[table_idx].append((name, types[idx]))

    pk_by_table = defaultdict(list)
    for idx in primary_keys:
        indices = idx if isinstance(idx, (list, tuple)) else [idx]
        for sub_idx in indices:
            table_idx, name = columns[sub_idx]
            pk_by_table[table_idx].append(name)

    fk_by_table = defaultdict(list)
    for src, dst in foreign_keys:
        src_table, src_col = columns[src]
        dst_table, dst_col = columns[dst]
        fk_by_table[src_table].append((src_col, tables[dst_table], dst_col))

    statements = []
    for table_idx, table_name in enumerate(tables):
        parts = [f"  {quote(name)} {ctype}" for name, ctype in by_table[table_idx]]
        if pk_by_table[table_idx]:
            cols = ", ".join(quote(c) for c in pk_by_table[table_idx])
            parts.append(f"  PRIMARY KEY ({cols})")
        for src_col, dst_table, dst_col in fk_by_table[table_idx]:
            parts.append(
                f"  FOREIGN KEY ({quote(src_col)}) REFERENCES {quote(dst_table)}({quote(dst_col)})"
            )
        statements.append(f"CREATE TABLE {quote(table_name)} (\n" + ",\n".join(parts) + "\n);")
    return "\n\n".join(statements)


def load_samples(root: Path) -> tuple[list[dict], dict]:
    dev = json.loads((root / "dev.json").read_text(encoding="utf-8"))
    tables = json.loads((root / "dev_tables.json").read_text(encoding="utf-8"))
    table_map = {entry["db_id"]: entry for entry in tables}
    db_dir = root / "dev_databases"

    samples, skipped = [], []
    for item in dev:
        db_id = item["db_id"]
        entry = table_map.get(db_id)
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        if entry is None or not db_path.exists():
            skipped.append({"question_id": item["question_id"], "db_id": db_id, "reason": "missing_db"})
            continue
        evidence = (item.get("evidence") or "").strip()
        evidence_block = f"\n### Evidence\n{evidence}\n" if evidence else ""
        samples.append(
            {
                "id": int(item["question_id"]),
                "db_id": db_id,
                "db_path": str(db_path),
                "question": item["question"],
                "evidence": evidence,
                "difficulty": item.get("difficulty", ""),
                "gold_sql": item["SQL"],
                "n_tables": len(entry["table_names_original"]),
                "n_columns": len([c for c in entry["column_names_original"] if c[0] >= 0]),
                "prompt": PROMPT.format(
                    schema=build_schema(entry), question=item["question"], evidence=evidence_block
                ),
            }
        )
    report = {"total": len(dev), "built": len(samples), "skipped": skipped}
    return samples, report


def stratified(samples: list[dict], limit: int, seed: int = 0) -> list[dict]:
    by_db = defaultdict(list)
    for sample in samples:
        by_db[sample["db_id"]].append(sample)
    rng = random.Random(seed)
    for items in by_db.values():
        rng.shuffle(items)
    chosen, cursor = [], 0
    while len(chosen) < limit:
        added = False
        for db_id in sorted(by_db):
            if cursor < len(by_db[db_id]) and len(chosen) < limit:
                chosen.append(by_db[db_id][cursor])
                added = True
        cursor += 1
        if not added:
            break
    return chosen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/dev_extract/dev_20240627")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root)
    samples, report = load_samples(root)
    out = Path(args.out)
    write_jsonl(out / "dev_all.jsonl", samples)
    subset = stratified(samples, args.limit) if args.limit else samples
    write_jsonl(out / "dev_200.jsonl", subset)

    report["subset"] = {"limit": args.limit, "size": len(subset)}
    report["difficulty"] = {
        key: sum(1 for s in subset if s["difficulty"] == key)
        for key in sorted({s["difficulty"] for s in subset})
    }
    report["dbs"] = sorted({s["db_id"] for s in subset})
    (out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("total", "built", "subset", "difficulty")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
