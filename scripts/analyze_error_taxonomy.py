"""Error taxonomy over evaluator records.

Reads `error_type` from the record top level when present, otherwise from
`score.error_type` (the rl_signal records keep it there). Keeps `gate.json`
semantics untouched; this is a diagnostic read-only aggregate.

Primary categories (mutually exclusive, in precedence order):
  infrastructure_error
  reward_unavailable
  correct
  invalid_<error_type>        e.g. invalid_no_such_column
  valid_but_wrong             valid SQL, wrong result set
  no_final_sql
  other_wrong
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value):
    if value is None or value == "" or value == "None" or value == "null":
        return None
    return str(value)


def error_type(record):
    top = normalized(record.get("error_type"))
    if top:
        return top
    return normalized((record.get("score") or {}).get("error_type"))


def primary_category(record):
    if record.get("infrastructure_error") or (record.get("score") or {}).get("infrastructure_error"):
        return "infrastructure_error"
    if record.get("reward_available") is False:
        return "reward_unavailable"
    if record.get("correct"):
        return "correct"
    kind = error_type(record)
    if kind:
        return f"invalid_{kind}"
    if record.get("valid"):
        return "valid_but_wrong"
    if not normalized(record.get("final_sql")):
        return "no_final_sql"
    return "other_wrong"


def tool_error_count(record):
    count = 0
    for event in record.get("executions") or []:
        result = event.get("result") if isinstance(event, dict) else None
        if isinstance(result, dict) and result.get("status") not in (None, "ok"):
            count += 1
    return count


def summarize(rows):
    total = len(rows)
    categories = Counter(primary_category(r) for r in rows)
    error_types = Counter(error_type(r) for r in rows)
    failure_origins = Counter(
        normalized((r.get("score") or {}).get("failure_origin")) for r in rows)
    finish = Counter(normalized(r.get("finish_reason")) for r in rows)
    valid_but_wrong = sum(1 for r in rows if r.get("valid") and not r.get("correct"))
    return {
        "records": total,
        "correct": sum(bool(r.get("correct")) for r in rows),
        "valid": sum(bool(r.get("valid")) for r in rows),
        "valid_but_wrong": valid_but_wrong,
        "no_such_column": error_types.get("no_such_column", 0),
        "infrastructure_error": categories.get("infrastructure_error", 0),
        "reward_unavailable": categories.get("reward_unavailable", 0),
        "categories": dict(categories.most_common()),
        "error_types": {str(k): v for k, v in error_types.most_common()},
        "failure_origins": {str(k): v for k, v in failure_origins.most_common()},
        "finish_reasons": {str(k): v for k, v in finish.most_common()},
        "tool_calls_total": sum(int(r.get("tool_calls") or 0) for r in rows),
        "schema_lookups_total": sum(int(r.get("schema_lookups") or 0) for r in rows),
        "sql_executes_total": sum(int(r.get("sql_executes") or 0) for r in rows),
        "invalid_actions_total": sum(int(r.get("invalid_actions") or 0) for r in rows),
        "tool_execution_errors": sum(tool_error_count(r) for r in rows),
        "latency_ms_mean": statistics.mean(
            float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None) if rows else None,
    }


def by_db(rows):
    groups = defaultdict(list)
    for record in rows:
        groups[str(record.get("db_id") or "unknown")].append(record)
    table = {}
    for db, items in groups.items():
        summary = summarize(items)
        table[db] = {key: summary[key] for key in (
            "records", "correct", "valid_but_wrong", "no_such_column",
            "infrastructure_error")}
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["records"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", default="")
    parser.add_argument("--focus-tasks", default="",
                        help="tasks.jsonl whose `id` values define a focus subset")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    rows = read_jsonl(args.records)
    result = {
        "records_path": str(Path(args.records).resolve()),
        "label": args.label,
        "overall": summarize(rows),
        "by_db": by_db(rows),
    }
    if args.focus_tasks:
        focus_ids = {str(row["id"]) for row in read_jsonl(args.focus_tasks)}
        focus = [row for row in rows if str(row.get("id")) in focus_ids]
        result["focus"] = {
            "tasks_path": str(Path(args.focus_tasks).resolve()),
            "task_ids": sorted(focus_ids),
            "records": summarize(focus),
            "per_question": {
                task_id: summarize([row for row in focus if str(row.get("id")) == task_id])
                for task_id in sorted(focus_ids)
            },
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")

    overall = result["overall"]
    lines = [f"# Error taxonomy{(' — ' + args.label) if args.label else ''}", "",
             f"Records: `{args.records}` ({overall['records']})", "",
             f"Correct: {overall['correct']} · valid-but-wrong: {overall['valid_but_wrong']} · "
             f"no_such_column: {overall['no_such_column']} · infra: {overall['infrastructure_error']}", "",
             "## Primary categories", "", "| category | count |", "|---|---:|"]
    for key, value in overall["categories"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Error types", "", "| error_type | count |", "|---|---:|"]
    for key, value in overall["error_types"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## By database (top 15 by records)", "",
              "| db | records | correct | valid_but_wrong | no_such_column | infra |",
              "|---|---:|---:|---:|---:|---:|"]
    for db, row in list(result["by_db"].items())[:15]:
        lines.append(f"| {db} | {row['records']} | {row['correct']} | "
                     f"{row['valid_but_wrong']} | {row['no_such_column']} | "
                     f"{row['infrastructure_error']} |")
    if "focus" in result:
        lines += ["", "## Focus questions", "",
                  "| question | records | correct | valid_but_wrong | no_such_column |",
                  "|---|---:|---:|---:|---:|"]
        for task_id, row in result["focus"]["per_question"].items():
            lines.append(f"| {task_id} | {row['records']} | {row['correct']} | "
                         f"{row['valid_but_wrong']} | {row['no_such_column']} |")
    lines.append("")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines))
    print(json.dumps(overall, ensure_ascii=False))


if __name__ == "__main__":
    main()
