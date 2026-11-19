"""Required metrics for the Qwen3-4B dev200 zero-shot probe (no training)."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--hbm", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = read_jsonl(a.records)
    n = len(rows)
    first_cases = first_ex = 0
    repair_cases = repair_ok = 0
    error_types = Counter()
    semantic_wrong = 0
    for r in rows:
        if r.get("valid") and not r.get("correct"):
            semantic_wrong += 1
        error_types[str(r.get("error_type"))] += 1
        first = r.get("first_sql")
        if first:
            first_cases += 1
            scored = score_sql(r["db_path"], first, r["gold_sql"], timeout=30)
            first_ex += int(bool(scored.get("correct")))
        executions = r.get("executions") or []
        if executions and (executions[0].get("result") or {}).get("status") != "ok":
            repair_cases += 1
            repair_ok += int(bool(r.get("correct")))
    hbm = []
    if a.hbm and Path(a.hbm).is_file():
        for line in Path(a.hbm).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    hbm.append(float(parts[1]))
                except ValueError:
                    pass
    out = {
        "n": n,
        "ex": sum(1 for r in rows if r.get("correct")) / max(n, 1),
        "valid": sum(1 for r in rows if r.get("valid")) / max(n, 1),
        "infra": sum(1 for r in rows if r.get("infrastructure_error")),
        "no_answer": sum(1 for r in rows if r.get("finish_reason") != "answer_submitted"),
        "first_sql_cases": first_cases,
        "first_sql_ex": first_ex / max(first_cases, 1),
        "repair_cases": repair_cases,
        "repair_rate": repair_ok / max(repair_cases, 1),
        "semantic_wrong": semantic_wrong,
        "no_such_column": sum(1 for r in rows if r.get("error_type") == "no_such_column"),
        "no_such_table": sum(1 for r in rows if r.get("error_type") == "no_such_table"),
        "lookup_rate": sum(1 for r in rows if (r.get("usage") or {}).get("looked_up")) / max(n, 1),
        "avg_tool_calls": sum(r.get("tool_calls", 0) for r in rows) / max(n, 1),
        "avg_schema_lookups": sum(r.get("schema_lookups", 0) for r in rows) / max(n, 1),
        "avg_sql_executes": sum(r.get("sql_executes", 0) for r in rows) / max(n, 1),
        "avg_response_tokens": sum(r.get("response_tokens", 0) for r in rows) / max(n, 1),
        "avg_observation_tokens": sum(r.get("observation_tokens", 0) for r in rows) / max(n, 1),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in rows) / max(n, 1),
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in rows)),
        "error_types": dict(error_types),
        "invalid_action_episodes": sum(1 for r in rows if (r.get("invalid_actions") or 0) > 0),
        "invalid_actions_total": sum(int(r.get("invalid_actions") or 0) for r in rows),
        "hbm_peak_pct": max(hbm) if hbm else None,
        "by_db": {
            db: {"n": len(rs), "correct": sum(1 for r in rs if r.get("correct"))}
            for db, rs in sorted({db: [r for r in rows if r.get("db_id") == db] for db in {r.get("db_id") for r in rows}}.items())
        },
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "ex", "valid", "first_sql_ex", "repair_rate",
                                         "no_such_column", "lookup_rate", "hbm_peak_pct")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
