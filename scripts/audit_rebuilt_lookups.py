"""Audit schema_lookup actions in the rebuilt E1-contract 3B SFT trajectories.

The rebuilt records store the E1 lookup result in `trajectory.tool_events`,
with `retrieval_metadata.matched_tables` (not the old `tables` list).
This script is read-only and writes the same kind of inventory/summary as
scripts/audit_sft_lookups.py, plus filtered CSVs for manual review.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

SQL_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def lookup_tables(result):
    if not isinstance(result, dict):
        return []
    md = result.get("retrieval_metadata") or {}
    tables = md.get("matched_tables") if isinstance(md, dict) else None
    if tables:
        return [str(x) for x in tables]
    tables = result.get("tables")
    if isinstance(tables, list):
        return [str(t.get("name")) for t in tables if isinstance(t, dict) and t.get("name")]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/m6/e1_contract_rebuild/sft_trajectories_e1.jsonl")
    p.add_argument("--out-dir", default="reports/m6/e1_contract_rebuild/audit")
    p.add_argument("--old-summary", default="reports/m6/sft_data_audit/lookup_summary.json")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    n_records = 0
    for rec in read_jsonl(a.data):
        n_records += 1
        traj = rec.get("trajectory") or {}
        segs = traj.get("segments") or []
        events = traj.get("tool_events") or []
        by_event = {int(e.get("tool_event_index", i)): e for i, e in enumerate(events)}
        for i, seg in enumerate(segs):
            if seg.get("kind") != "action" or seg.get("action") != "schema_lookup":
                continue
            event = by_event.get(int(seg.get("tool_event_index", -1)))
            result = (event or {}).get("result") or {}
            status = str(result.get("status"))
            tables = lookup_tables(result)
            obs_text = str(result.get("observation") or "")
            empty = status == "ok" and not tables and "Tables:" not in obs_text
            error = status != "ok"
            nxt = None
            for j in range(i + 2, len(segs)):
                if segs[j].get("kind") == "action":
                    nxt = segs[j]
                    break
            nxt_text = (nxt or {}).get("text") or ""
            m = re.match(r"<([a-z_]+)>", nxt_text.strip())
            nxt_tag = m.group(1) if m else None
            sql_tables = []
            if nxt_tag in ("sql_execute", "answer"):
                sql_tables = sorted({x.lower() for x in SQL_TABLE_RE.findall(nxt_text)})
            obs_lower = {x.lower() for x in tables}
            used = None
            if sql_tables:
                used = any(t in obs_lower for t in sql_tables)
            rows.append({
                "trajectory_id": rec.get("id"),
                "original_trajectory_id": rec.get("original_trajectory_id"),
                "question_id": rec.get("question_id"),
                "db_id": rec.get("db_id"),
                "source_type": rec.get("source_type"),
                "error_types": rec.get("error_types"),
                "final_ex": rec.get("final_ex"),
                "action_index": sum(1 for s in segs[:i + 1] if s.get("kind") == "action"),
                "supervise": seg.get("supervise"),
                "lookup_args": seg.get("arguments"),
                "lookup_text": seg.get("text"),
                "observation_status": status,
                "observation_error_type": result.get("error_type"),
                "observation_message": (result.get("message") or "")[:300],
                "observation_tables": tables,
                "observation_n_tables": len(tables),
                "observation_text_chars": len(obs_text),
                "lookup_empty": empty,
                "lookup_error": error,
                "lookup_bad": empty or error,
                "next_action_tag": nxt_tag,
                "next_action_text": nxt_text[:1000],
                "sql_tables": sql_tables,
                "sql_uses_observed_table": used,
            })

    fields = ["trajectory_id", "original_trajectory_id", "question_id", "db_id", "source_type",
              "error_types", "action_index", "supervise", "lookup_args", "lookup_text",
              "observation_status", "observation_error_type", "observation_message",
              "observation_tables", "observation_n_tables", "lookup_empty", "lookup_error",
              "lookup_bad", "next_action_tag", "next_action_text", "sql_tables",
              "sql_uses_observed_table"]
    with (out / "lookup_inventory.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "lookup_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict, list))
                        else r.get(k) for k in fields})
    for name, filt in (
        ("lookup_supervised_empty.csv", lambda r: r["lookup_empty"] and r["supervise"] is True),
        ("lookup_bad.csv", lambda r: r["lookup_bad"]),
        ("lookup_sql_table_mismatch.csv", lambda r: r["sql_uses_observed_table"] is False),
    ):
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                if filt(r):
                    w.writerow({k: json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict, list))
                                else r.get(k) for k in fields})

    supervised = [r for r in rows if r.get("supervise") is True]
    by_source = {}
    for r in rows:
        src = str(r.get("source_type"))
        by_source.setdefault(src, {"n": 0, "bad": 0, "empty": 0, "error": 0})
        by_source[src]["n"] += 1
        by_source[src]["bad"] += int(bool(r["lookup_bad"]))
        by_source[src]["empty"] += int(bool(r["lookup_empty"]))
        by_source[src]["error"] += int(bool(r["lookup_error"]))
    summary = {
        "records": n_records,
        "lookups": len(rows),
        "supervised_lookups": len(supervised),
        "lookup_empty": sum(bool(r["lookup_empty"]) for r in rows),
        "lookup_error": sum(bool(r["lookup_error"]) for r in rows),
        "lookup_bad": sum(bool(r["lookup_bad"]) for r in rows),
        "supervised_empty": sum(bool(r["lookup_empty"]) for r in supervised),
        "supervised_error": sum(bool(r["lookup_error"]) for r in supervised),
        "supervised_bad": sum(bool(r["lookup_bad"]) for r in supervised),
        "observation_status": dict(Counter(str(r["observation_status"]) for r in rows)),
        "next_action_tag": dict(Counter(str(r["next_action_tag"]) for r in rows)),
        "sql_uses_observed_table": dict(Counter(str(r["sql_uses_observed_table"]) for r in rows)),
        "by_source_type": by_source,
    }
    if Path(a.old_summary).is_file():
        old = json.loads(Path(a.old_summary).read_text(encoding="utf-8"))
        summary["old_contract_baseline"] = {
            "lookups": old.get("lookups"), "supervised_lookups": old.get("supervised_lookups"),
            "lookup_empty": old.get("lookup_empty"), "lookup_error": old.get("lookup_error"),
            "lookup_bad": old.get("lookup_bad"), "supervised_bad": old.get("supervised_bad"),
            "sql_uses_observed_table": old.get("sql_uses_observed_table"),
        }
    (out / "lookup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 重建后 E1 契约 schema_lookup 审计", ""]
    lines.append(f"- 记录 {n_records}；lookup {len(rows)}；被监督 lookup {len(supervised)}")
    lines.append(f"- empty {summary['lookup_empty']}；error {summary['lookup_error']}；bad {summary['lookup_bad']}")
    lines.append(f"- 被监督 empty {summary['supervised_empty']}；被监督 bad {summary['supervised_bad']}")
    lines.append(f"- observation status：{summary['observation_status']}")
    lines.append(f"- SQL 是否用到 observation 表：{summary['sql_uses_observed_table']}")
    if "old_contract_baseline" in summary:
        lines.append(f"- 旧契约基线：{summary['old_contract_baseline']}")
    lines.append("")
    lines.append("## 按 source_type")
    for src, x in by_source.items():
        lines.append(f"- {src}: n={x['n']} bad={x['bad']} empty={x['empty']} error={x['error']}")
    (out / "lookup_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("records", "lookups", "supervised_lookups",
                                              "lookup_empty", "lookup_error", "lookup_bad",
                                              "supervised_empty", "supervised_bad")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
