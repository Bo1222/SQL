"""M6-02: audit whether the frozen SFT policy still behaves under the E1 runtime.

Reads existing M5 traces (no model, no NPU). Answers the six questions in
docs/TASKS_M6.md and returns an explicit "does this trigger adaptation SFT" verdict.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tables_of(payload):
    """Tables exposed by the first retrieval, for legacy and unified payloads."""
    if not isinstance(payload, dict):
        return set()
    meta = payload.get("retrieval_metadata") or {}
    if meta.get("matched_tables"):
        return {str(t).lower() for t in meta["matched_tables"]}
    return {str(t.get("name", "")).lower() for t in payload.get("tables", []) if t.get("name")}


def columns_of(payload):
    if not isinstance(payload, dict):
        return set()
    meta = payload.get("retrieval_metadata") or {}
    names = {str(c.get("column", "")).lower() for c in meta.get("matched_columns", [])}
    for table in payload.get("tables", []):
        names |= {str(c.get("name", "")).lower() for c in table.get("columns", [])}
    return {n for n in names if n}


def sql_mentions(sql, names):
    if not sql:
        return False
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql.lower()))
    return bool(tokens & names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="reports/m5/traces_unified/episodes.jsonl")
    p.add_argument("--data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--out", default="reports/m6/sft_compat_audit.json")
    p.add_argument("--format-error-threshold", type=float, default=0.02)
    p.add_argument("--repair-floor", type=float, default=0.20)
    a = p.parse_args()

    episodes = read_jsonl(a.trace)
    if not episodes:
        raise SystemExit(f"no episodes at {a.trace}")
    gold = {row["id"]: row for row in read_jsonl(a.data)}

    n = len(episodes)
    invalid = sum(1 for e in episodes if e.get("finish_reason") == "invalid_action")
    no_answer = sum(1 for e in episodes if e.get("finish_reason") != "answer_submitted")
    lookups = [e for e in episodes if (e.get("schema_lookups") or 0) > 0]
    first_lookup_payloads = []
    for e in episodes:
        first = next((ev for ev in e.get("tool_events", []) if ev.get("action") == "schema_lookup"), None)
        first_lookup_payloads.append(first.get("result") if first else None)
    empty_retrieval = sum(1 for payload in first_lookup_payloads
                          if payload is not None and payload.get("status") == "ok" and not tables_of(payload))

    utilized = 0
    for e, payload in zip(episodes, first_lookup_payloads):
        names = tables_of(payload) | columns_of(payload)
        if names and sql_mentions(e.get("final_sql"), names):
            utilized += 1

    # repair: first execution failed (tool error) but the episode ended correct
    repair_den = repair_ok = 0
    for e in episodes:
        executions = e.get("executions") or []
        if not executions:
            continue
        first = executions[0].get("result") or {}
        if first.get("status") == "error":
            repair_den += 1
            if e.get("correct"):
                repair_ok += 1

    correct = sum(1 for e in episodes if e.get("correct"))
    valid = sum(1 for e in episodes if e.get("valid"))
    invalid_rate = invalid / n
    repair_rate = repair_ok / repair_den if repair_den else None

    checks = {
        "format_error_rate": {"value": round(invalid_rate, 4),
                              "threshold": f"<= {a.format_error_threshold}",
                              "pass": invalid_rate <= a.format_error_threshold},
        "empty_retrieval_rate": {"value": round(empty_retrieval / max(len(lookups), 1), 4),
                                 "threshold": "<= 0.05", "pass": empty_retrieval / max(len(lookups), 1) <= 0.05},
        "repair_rate": {"value": round(repair_rate, 4) if repair_rate is not None else None,
                        "threshold": f">= {a.repair_floor}", "pass": (repair_rate or 0) >= a.repair_floor},
    }
    triggered = [name for name, check in checks.items() if not check["pass"]]

    report = {
        "format": "m6-02-sft-compat-audit-v1",
        "trace": a.trace,
        "n": n,
        "model": "seed_0/step_600 (merged)",
        "runtime": "E1 unified backend, autonomous lookup, greedy",
        "metrics": {
            "ex": round(correct / n, 4),
            "valid_rate": round(valid / n, 4),
            "no_answer_rate": round(no_answer / n, 4),
            "invalid_action_episodes": invalid,
            "lookup_rate": round(len(lookups) / n, 4),
            "empty_retrieval_rate": round(empty_retrieval / max(len(lookups), 1), 4),
            "retrieval_utilization_rate": round(utilized / n, 4),
            "repair_denominator": repair_den,
            "repair_correct": repair_ok,
            "avg_tool_calls": round(sum(e.get("tool_calls", 0) for e in episodes) / n, 2),
            "avg_schema_lookups": round(sum(e.get("schema_lookups", 0) for e in episodes) / n, 2),
            "avg_sql_executes": round(sum(e.get("sql_executes", 0) for e in episodes) / n, 2),
        },
        "checks": checks,
        "decision": {
            "adaptation_sft_triggered": bool(triggered),
            "failed_checks": triggered,
            "rule": "trigger adaptation SFT only if format error rate > 2% or repair rate < 20%",
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=1))
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
