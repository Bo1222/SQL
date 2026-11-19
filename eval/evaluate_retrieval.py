"""Offline retrieval-only comparison (no model): legacy vs unified backend.

Reports Table/Column/Join-Key recall@K, catalog match, empty/fallback rate,
observation tokens and retrieval latency on the same questions.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.backends import build_backend  # noqa: E402
from eval.gold_extraction import extract_requirements  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def usage_from_payload(payload, requirements):
    tables, columns, relationships = set(), set(), set()
    if payload.get("retrieval_metadata"):
        meta = payload["retrieval_metadata"]
        tables = {t.lower() for t in meta.get("matched_tables", [])}
        columns = {f"{c['table']}.{c['column']}".lower() for c in meta.get("matched_columns", [])}
        relationships = {(r.get("from_table", "").lower(), r.get("from_column", "").lower(),
                          r.get("to_table", "").lower(), r.get("to_column", "").lower())
                         for r in (meta.get("relationships") or []) if isinstance(r, dict)}
    else:
        for table in payload.get("tables", []):
            tables.add(table.get("name", "").lower())
            for column in table.get("columns", []):
                columns.add(f"{table['name']}.{column['name']}".lower())
            for fk in table.get("foreign_keys", []):
                relationships.add((table.get("name", "").lower(), fk.get("column", "").lower(),
                                   fk.get("target_table", "").lower(), fk.get("target_column", "").lower()))
    gold_tables = set(requirements["tables"])
    gold_columns = set(requirements["columns"])
    gold_joins = set()
    for item in requirements["join_keys"]:
        left, right = item.split("=")
        a, ca = left.split(".")
        b, cb = right.split(".")
        gold_joins.add((a, b, ca, cb))
    join_hits = 0
    for a, b, ca, cb in gold_joins:
        if (a, ca, b, cb) in relationships or (b, cb, a, ca) in relationships:
            join_hits += 1
    return {
        "table_recall": len(tables & gold_tables) / max(len(gold_tables), 1),
        "column_recall": len(columns & gold_columns) / max(len(gold_columns), 1),
        "join_recall": (join_hits / len(gold_joins)) if gold_joins else 1.0,
        "tables": sorted(tables), "columns": sorted(columns),
        "empty": len(tables) == 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--catalog-dir", default="artifacts/catalog")
    p.add_argument("--observation-budget", type=int, default=1024)
    p.add_argument("--out", default="reports/m2/retrieval_comparison.json")
    p.add_argument("--trace", default="")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/bird_agent_full_sft_20261026/seed_0/step_600")
    rows = read_jsonl(a.data)[:a.limit]
    results = {}
    for backend_name in ("legacy", "unified"):
        records = []
        for row in rows:
            requirements = extract_requirements(row["gold_sql"], row["db_path"])
            backend = build_backend(backend_name, row, tokenizer=tokenizer, catalog_dir=a.catalog_dir,
                                    token_budget=a.observation_budget)
            query = f"{row['question']} {row.get('evidence', '')}".strip()
            started = time.monotonic()
            try:
                payload = backend({"query": query})
            except Exception as exc:  # a backend crash is a data point, not a stop
                payload = {"status": "error", "error": str(exc)[:300]}
            latency = (time.monotonic() - started) * 1000
            usage = usage_from_payload(payload, requirements) if payload.get("status") == "ok" else {
                "table_recall": 0.0, "column_recall": 0.0, "join_recall": 0.0, "tables": [], "columns": [],
                "empty": True}
            observation = payload.get("observation") or json.dumps(payload)[:2000]
            tokens = len(tokenizer.encode(observation, add_special_tokens=False))
            records.append({"id": row["id"], "db_id": row["db_id"], "status": payload.get("status"),
                            "error": payload.get("error") or payload.get("message"),
                            "latency_ms": round(latency, 2), "observation_tokens": tokens,
                            **usage, "requirements": requirements,
                            "fallback_used": (payload.get("retrieval_metadata") or {}).get("fallback_used", False)})
        ok = [r for r in records if r["status"] == "ok"]
        results[backend_name] = {
            "n": len(records),
            "error_rate": 1 - len(ok) / max(len(records), 1),
            "table_recall": st.mean(r["table_recall"] for r in records),
            "column_recall": st.mean(r["column_recall"] for r in records),
            "join_recall": st.mean(r["join_recall"] for r in records),
            "empty_rate": sum(1 for r in records if r["empty"]) / max(len(records), 1),
            "fallback_rate": sum(1 for r in records if r["fallback_used"]) / max(len(records), 1),
            "observation_tokens": st.mean(r["observation_tokens"] for r in records),
            "latency_ms": st.mean(r["latency_ms"] for r in records),
            "records": records,
        }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {name: {k: v for k, v in data.items() if k != "records"} for name, data in results.items()}
    out.write_text(json.dumps({"summary": summary, "detail": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.trace:
        Path(a.trace).parent.mkdir(parents=True, exist_ok=True)
        Path(a.trace).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
