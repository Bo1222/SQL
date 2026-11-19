"""Read-only audit of schema_lookup actions in the original 3B SFT trajectories.

Input: artifacts/datasets/sft_trajectories.jsonl (4,471 records)
Output: reports/m6/sft_data_audit/lookup_inventory.{jsonl,csv}, lookup_summary.json/.md

It does NOT modify the training data. It extracts every schema_lookup action,
the following observation, and the next action, then classifies:
  - malformed action JSON / missing query
  - observation status error / missing / empty tables
  - whether the lookup was supervised
  - whether the later SQL uses a table returned by the lookup

This is an audit aid for manual review, not a data-cleaning pipeline.
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ACTION_RE = re.compile(r"<schema_lookup>(.*?)</schema_lookup>", re.S)
OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.S)
TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_action(text):
    m = ACTION_RE.search(text or "")
    if not m:
        return {"parse_ok": False, "kind": "no_tag", "args": None, "reason": "no_schema_lookup_tag"}
    body = m.group(1).strip()
    try:
        args = json.loads(body)
    except Exception as exc:
        return {"parse_ok": False, "kind": "bad_json", "args": None, "reason": str(exc)[:160]}
    if not isinstance(args, dict):
        return {"parse_ok": False, "kind": "not_object", "args": None, "reason": "json_not_object"}
    if "query" in args:
        kind = "query"
    elif "table" in args:
        kind = "table"
    elif "column" in args:
        kind = "column"
    else:
        kind = "other"
    return {"parse_ok": True, "kind": kind, "args": args, "reason": None}


def parse_observation(text):
    m = OBS_RE.search(text or "")
    if not m:
        return {"parse_ok": False, "status": "missing", "error_type": None, "message": None,
                "tables": [], "n_tables": 0, "n_columns": 0}
    body = m.group(1).strip()
    try:
        data = json.loads(body)
    except Exception as exc:
        return {"parse_ok": False, "status": "bad_json", "error_type": None, "message": str(exc)[:160],
                "tables": [], "n_tables": 0, "n_columns": 0}
    tables = data.get("tables") or []
    if not isinstance(tables, list):
        tables = []
    names = [str(t.get("name")) for t in tables if isinstance(t, dict) and t.get("name")]
    n_cols = sum(len(t.get("columns") or []) for t in tables if isinstance(t, dict))
    return {"parse_ok": True, "status": str(data.get("status")),
            "error_type": data.get("error_type"), "message": data.get("message"),
            "tables": names, "n_tables": len(tables), "n_columns": n_cols}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/datasets/sft_trajectories.jsonl")
    p.add_argument("--out-dir", default="reports/m6/sft_data_audit")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    n_records = 0
    for rec in read_jsonl(a.data):
        n_records += 1
        segs = (rec.get("trajectory") or {}).get("segments") or []
        action_index = 0
        for i, seg in enumerate(segs):
            if seg.get("kind") != "action":
                continue
            action_index += 1
            text = seg.get("text") or ""
            if not text.strip().startswith("<schema_lookup>"):
                continue
            act = parse_action(text)
            obs_seg = segs[i + 1] if i + 1 < len(segs) and segs[i + 1].get("kind") == "observation" else None
            obs = parse_observation(obs_seg.get("text") if obs_seg else "")
            nxt = None
            for j in range(i + 2, len(segs)):
                if segs[j].get("kind") == "action":
                    nxt = segs[j]
                    break
            nxt_text = (nxt or {}).get("text") or ""
            nxt_tag = None
            m = re.match(r"<([a-z_]+)>", nxt_text.strip())
            if m:
                nxt_tag = m.group(1)
            sql_tables = []
            if nxt_tag in ("sql_execute", "answer"):
                sql_tables = sorted({x.lower() for x in TABLE_RE.findall(nxt_text)})
            obs_tables_lower = {x.lower() for x in obs["tables"]}
            observed = None
            if sql_tables:
                observed = any(t in obs_tables_lower for t in sql_tables)
            malformed = not act["parse_ok"]
            error = (not obs["parse_ok"]) or obs["status"] != "ok"
            empty = obs["parse_ok"] and obs["status"] == "ok" and obs["n_tables"] == 0
            bad = malformed or error or empty
            row = {
                "trajectory_id": rec.get("trajectory_id"),
                "question_id": rec.get("question_id"),
                "db_id": rec.get("db_id"),
                "source_type": rec.get("source_type"),
                "error_types": rec.get("error_types"),
                "final_ex": rec.get("final_ex"),
                "action_index": action_index,
                "segment_index": i,
                "supervise": seg.get("supervise"),
                "lookup_kind": act["kind"],
                "lookup_parse_ok": act["parse_ok"],
                "lookup_args": act["args"],
                "lookup_reason": act["reason"],
                "lookup_text": text[:1000],
                "observation_status": obs["status"],
                "observation_error_type": obs["error_type"],
                "observation_message": (obs["message"] or "")[:300],
                "observation_tables": obs["tables"],
                "observation_n_tables": obs["n_tables"],
                "observation_n_columns": obs["n_columns"],
                "observation_parse_ok": obs["parse_ok"],
                "observation_missing": obs["status"] == "missing",
                "lookup_malformed": malformed,
                "lookup_error": error,
                "lookup_empty": empty,
                "lookup_bad": bad,
                "next_action_tag": nxt_tag,
                "next_action_text": nxt_text[:1000],
                "sql_tables": sql_tables,
                "sql_uses_observed_table": observed,
            }
            rows.append(row)

    with (out / "lookup_inventory.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["trajectory_id", "question_id", "db_id", "source_type", "action_index", "supervise",
              "lookup_kind", "lookup_parse_ok", "lookup_reason", "lookup_args", "observation_status",
              "observation_error_type", "observation_message", "observation_n_tables",
              "observation_n_columns", "lookup_malformed", "lookup_error", "lookup_empty",
              "lookup_bad", "next_action_tag", "sql_tables", "sql_uses_observed_table"]
    with (out / "lookup_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list))
                        else row.get(k) for k in fields})

    def by(key):
        return dict(Counter(str(r.get(key)) for r in rows))

    supervised = [r for r in rows if r.get("supervise") is True]
    summary = {
        "records": n_records,
        "lookups": len(rows),
        "supervised_lookups": len(supervised),
        "unsupervised_lookups": sum(1 for r in rows if r.get("supervise") is not True),
        "lookup_kind": by("lookup_kind"),
        "lookup_parse_ok": by("lookup_parse_ok"),
        "observation_status": by("observation_status"),
        "observation_error_type": by("observation_error_type"),
        "lookup_malformed": sum(bool(r["lookup_malformed"]) for r in rows),
        "lookup_error": sum(bool(r["lookup_error"]) for r in rows),
        "lookup_empty": sum(bool(r["lookup_empty"]) for r in rows),
        "lookup_bad": sum(bool(r["lookup_bad"]) for r in rows),
        "observation_missing": sum(bool(r["observation_missing"]) for r in rows),
        "supervised_bad": sum(bool(r["lookup_bad"]) for r in supervised),
        "supervised_error": sum(bool(r["lookup_error"]) for r in supervised),
        "supervised_empty": sum(bool(r["lookup_empty"]) for r in supervised),
        "supervised_malformed": sum(bool(r["lookup_malformed"]) for r in supervised),
        "next_action_tag": by("next_action_tag"),
        "sql_uses_observed_table": by("sql_uses_observed_table"),
        "by_source_type": {
            str(src): {
                "n": len(group),
                "bad": sum(bool(r["lookup_bad"]) for r in group),
                "error": sum(bool(r["lookup_error"]) for r in group),
                "empty": sum(bool(r["lookup_empty"]) for r in group),
            }
            for src, group in _group(rows, "source_type").items()
        },
        "top_db": Counter(str(r.get("db_id")) for r in rows).most_common(20),
        "top_error_types": Counter(str(r.get("error_types")) for r in rows if r.get("lookup_bad")).most_common(20),
        "query_len": _stats([len(str((r.get("lookup_args") or {}).get("query") or ""))
                             for r in rows if r.get("lookup_kind") == "query"]),
    }
    (out / "lookup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 3B SFT schema_lookup 动作审计（只读）", ""]
    lines.append(f"- 记录数：**{n_records}**；lookup 总数：**{len(rows)}**；被监督 lookup：**{len(supervised)}**")
    lines.append(f"- malformed：**{summary['lookup_malformed']}**；observation error：**{summary['lookup_error']}**；"
                 f"empty tables：**{summary['lookup_empty']}**；任一问题：**{summary['lookup_bad']}**")
    lines.append(f"- 被监督 lookup 中有问题：**{summary['supervised_bad']}**"
                 f"（error {summary['supervised_error']} / empty {summary['supervised_empty']} / malformed {summary['supervised_malformed']}）")
    lines.append(f"- observation status：{summary['observation_status']}")
    lines.append(f"- lookup kind：{summary['lookup_kind']}")
    lines.append(f"- next action：{summary['next_action_tag']}")
    lines.append(f"- SQL 是否用到 observation 表：{summary['sql_uses_observed_table']}")
    lines.append("")
    lines.append("## 按 source_type")
    for src, x in summary["by_source_type"].items():
        lines.append(f"- {src}: n={x['n']} bad={x['bad']} error={x['error']} empty={x['empty']}")
    lines.append("")
    lines.append("## 有问题的 lookup 示例（最多 10 条）")
    bad_rows = [r for r in rows if r["lookup_bad"]][:10]
    for r in bad_rows:
        lines.append(f"- `{r['trajectory_id']}` {r['db_id']} supervise={r['supervise']} "
                     f"kind={r['lookup_kind']} status={r['observation_status']} "
                     f"tables={r['observation_n_tables']} next={r['next_action_tag']}")
        lines.append(f"  - action: `{r['lookup_text'][:300]}`")
        if r.get("observation_message"):
            lines.append(f"  - obs: `{r['observation_message'][:200]}`")
    (out / "lookup_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("records", "lookups", "supervised_lookups",
                                              "lookup_malformed", "lookup_error", "lookup_empty",
                                              "lookup_bad", "supervised_bad")}, ensure_ascii=False))


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[str(r.get(key))].append(r)
    return out


def _stats(vals):
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    n = len(vals)
    return {"n": n, "mean": round(sum(vals) / n, 1), "p50": vals[n // 2],
            "p90": vals[int(n * 0.9)], "max": max(vals)}


if __name__ == "__main__":
    main()
