"""Capability metrics for one dev200 evaluation run (SFT baseline or an RL checkpoint).

Adds First-SQL-EX and Repair-Success on top of evaluate_agent's summary by
re-scoring the first executed SQL against gold, reusing the training reward.
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = read_jsonl(a.records)
    gold = {row["id"]: row for row in read_jsonl(a.data)}
    per = []
    for record in rows:
        gold_row = gold.get(record["id"])
        if not gold_row:
            continue
        executions = record.get("executions") or []
        first = executions[0] if executions else None
        first_correct = False
        first_status = None
        if first is not None:
            first_status = (first.get("result") or {}).get("status")
            scored = score_sql(gold_row["db_path"], first.get("sql"), gold_row["gold_sql"], timeout=30)
            first_correct = bool(scored.get("correct"))
        error_type = None
        for execution in executions:
            error = (execution.get("result") or {}).get("error_type")
            if error:
                error_type = error
                break
        per.append({
            "id": record["id"],
            "db_id": record.get("db_id"),
            "correct": bool(record.get("correct")),
            "valid": bool(record.get("valid")),
            "first_sql_present": first is not None,
            "first_sql_correct": first_correct,
            "first_exec_status": first_status,
            "first_error_type": error_type,
            "repair_case": bool(first is not None and first_status == "error"),
            "repair_success": bool(first is not None and first_status == "error" and record.get("correct")),
            "schema_error": bool(error_type == "schema_error"),
            "lookups": record.get("schema_lookups", 0),
            "tool_calls": record.get("tool_calls", 0),
            "response_tokens": record.get("response_tokens", 0),
            "latency_ms": record.get("latency_ms", 0),
        })

    n = len(per)
    repair_den = sum(1 for r in per if r["repair_case"])
    repair_ok = sum(1 for r in per if r["repair_success"])
    summary = {
        "n": n,
        "final_ex": round(sum(1 for r in per if r["correct"]) / max(n, 1), 4),
        "valid_rate": round(sum(1 for r in per if r["valid"]) / max(n, 1), 4),
        "first_sql_ex": round(sum(1 for r in per if r["first_sql_correct"]) / max(n, 1), 4),
        "first_sql_present_rate": round(sum(1 for r in per if r["first_sql_present"]) / max(n, 1), 4),
        "repair_cases": repair_den,
        "repair_success": repair_ok,
        "repair_rate": round(repair_ok / repair_den, 4) if repair_den else None,
        "schema_error_episodes": sum(1 for r in per if r["schema_error"]),
        "lookup_rate": round(sum(1 for r in per if r["lookups"] > 0) / max(n, 1), 4),
        "avg_tool_calls": round(st.mean(r["tool_calls"] for r in per), 2),
        "avg_response_tokens": round(st.mean(r["response_tokens"] for r in per), 1),
        "avg_latency_ms": round(st.mean(r["latency_ms"] for r in per), 1),
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in rows)),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_question": per}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
