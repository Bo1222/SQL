"""Audit schema_lookup usage in evaluator records.

Distinguishes:
  query_only  -> {"query": ...}
  table_only  -> {"table": ...}
  table_column-> {"table": ..., "column": ...}
  other       -> neither

Reads the actual `tool_events` shape:
  {"action": "schema_lookup", "arguments": {...}, "result": {...}}
where `result.retrieval_metadata` has:
  matched_tables (list), matched_columns (list of dicts), selected_columns (int,
  exact path only), table_hint/column_hint (exact path only), retrieval_version.
There is no `inspection` field; it is derived from arguments or the hint pair.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def column_name(entry):
    if isinstance(entry, dict):
        return str(entry.get("column") or "").lower()
    return str(entry or "").lower()


def analyze_record(record):
    arm = record.get("arm") or "all"
    events = [e for e in (record.get("tool_events") or [])
              if e.get("action") == "schema_lookup"]
    rows = []
    for event in events:
        args = parse_args(event.get("arguments"))
        table = str(args.get("table") or "")
        column = str(args.get("column") or "")
        query = str(args.get("query") or "")
        if table and column:
            kind = "table_column"
        elif table:
            kind = "table_only"
        elif query:
            kind = "query_only"
        else:
            kind = "other"
        result = event.get("result") or {}
        meta = result.get("retrieval_metadata") or {}
        matched_columns = meta.get("matched_columns") or []
        matched_tables = [str(t).lower() for t in (meta.get("matched_tables") or [])]
        selected = meta.get("selected_columns")
        selected = selected if isinstance(selected, int) else 0
        table_hint = meta.get("table_hint")
        column_hint = meta.get("column_hint")
        inspection = bool((table and column) or (table_hint and column_hint))
        if kind == "table_column" and column:
            hit = any(column_name(c) == column.lower() for c in matched_columns)
        elif kind == "table_only" and table:
            hit = table.lower() in matched_tables
        else:
            hit = None
        observation = str(result.get("observation") or "")
        truncated = bool(meta.get("truncated")) or "truncated" in observation.lower()
        rows.append({
            "kind": kind, "table": table, "column": column, "query": query,
            "status": result.get("status"), "ok": result.get("status") == "ok",
            "inspection": inspection, "hit": hit,
            "selected_columns": selected,
            "observation_chars": len(observation), "truncated": truncated,
        })
    return {
        "id": record.get("id"), "arm": arm,
        "lookups": len(rows),
        "query_only": sum(r["kind"] == "query_only" for r in rows),
        "table_only": sum(r["kind"] == "table_only" for r in rows),
        "table_column": sum(r["kind"] == "table_column" for r in rows),
        "other": sum(r["kind"] == "other" for r in rows),
        "exact_records": int(any(r["kind"] == "table_column" for r in rows)),
        "exact_ok": sum(r["kind"] == "table_column" and r["ok"] for r in rows),
        "exact_inspection": sum(r["kind"] == "table_column" and r["inspection"] for r in rows),
        "exact_hit": sum(r["kind"] == "table_column" and bool(r["hit"]) for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "observation_chars": [r["observation_chars"] for r in rows],
        "events": rows,
    }


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["arm"]].append(record)
    out = {}
    for arm, rows in sorted(groups.items()):
        exact = [r for row in rows for r in row["events"] if r["kind"] == "table_column"]
        all_events = [r for row in rows for r in row["events"]]
        exact_questions = [row for row in rows if row["exact_records"]]
        out[arm] = {
            "records": len(rows),
            "lookups": sum(row["lookups"] for row in rows),
            "query_only": sum(row["query_only"] for row in rows),
            "table_only": sum(row["table_only"] for row in rows),
            "table_column": len(exact),
            "other": sum(row["other"] for row in rows),
            "exact_question_count": len(exact_questions),
            "exact_question_rate": len(exact_questions) / len(rows) if rows else None,
            "exact_ok_rate": (sum(r["ok"] for r in exact) / len(exact)) if exact else None,
            "exact_inspection_rate": (sum(r["inspection"] for r in exact) / len(exact)) if exact else None,
            "exact_hit_rate": (sum(bool(r["hit"]) for r in exact) / len(exact)) if exact else None,
            "exact_selected_columns_positive": sum(r["selected_columns"] > 0 for r in exact),
            "exact_selected_columns_rate": (sum(r["selected_columns"] > 0 for r in exact) / len(exact))
            if exact else None,
            "truncated_events": sum(r["truncated"] for r in all_events),
            "observation_chars_max": max((r["observation_chars"] for r in all_events), default=0),
            "observation_chars_mean": (statistics.mean(r["observation_chars"] for r in all_events)
                                       if all_events else None),
        }
    return out


def per_question(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["id"]].append(record)
    out = {}
    for question_id, rows in sorted(groups.items()):
        out[str(question_id)] = {
            "records": len(rows),
            "lookups": sum(row["lookups"] for row in rows),
            "query_only": sum(row["query_only"] for row in rows),
            "table_only": sum(row["table_only"] for row in rows),
            "table_column": sum(row["table_column"] for row in rows),
            "exact_question": int(any(row["exact_records"] for row in rows)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()
    records = [analyze_record(r) for r in read_jsonl(args.records)]
    result = {
        "records_path": str(Path(args.records).resolve()),
        "aggregate": aggregate(records),
        "per_question": per_question(records),
        "per_record": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    lines = ["# schema_lookup usage audit", "",
             f"Records: `{args.records}`", "",
             "| arm | records | lookups | query_only | table_only | table_column | exact questions | exact ok | exact inspection | exact selected | exact hit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, row in result["aggregate"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {arm} | {row['records']} | {row['lookups']} | {row['query_only']} | "
                     f"{row['table_only']} | {row['table_column']} | {row['exact_question_count']} | "
                     f"{fmt(row['exact_ok_rate'])} | {fmt(row['exact_inspection_rate'])} | "
                     f"{fmt(row['exact_selected_columns_rate'])} | {fmt(row['exact_hit_rate'])} |")
    lines += ["", f"Max observation chars: {max((r['observation_chars_max'] for r in result['aggregate'].values()), default=0)}", ""]
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines))
    print(json.dumps(result["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
