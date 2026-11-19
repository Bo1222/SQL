"""Summarize a Qwen3-4B zero-shot eval JSONL plus HBM samples (stdlib only)."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--hbm", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = read_jsonl(a.records)
    n = len(rows)
    summary = json.loads(Path(a.summary).read_text(encoding="utf-8")) if a.summary and Path(a.summary).is_file() else {}
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
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in rows)),
        "error_types": dict(Counter(r.get("error_type") for r in rows)),
        "invalid_action_episodes": sum(1 for r in rows if (r.get("invalid_actions") or 0) > 0),
        "invalid_actions_total": sum(int(r.get("invalid_actions") or 0) for r in rows),
        "lookup_rate": sum(1 for r in rows if (r.get("usage") or {}).get("looked_up")) / max(n, 1),
        "avg_tool_calls": sum(r.get("tool_calls", 0) for r in rows) / max(n, 1),
        "avg_schema_lookups": sum(r.get("schema_lookups", 0) for r in rows) / max(n, 1),
        "avg_sql_executes": sum(r.get("sql_executes", 0) for r in rows) / max(n, 1),
        "avg_response_tokens": sum(r.get("response_tokens", 0) for r in rows) / max(n, 1),
        "avg_observation_tokens": sum(r.get("observation_tokens", 0) for r in rows) / max(n, 1),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in rows) / max(n, 1),
        "hbm_samples": len(hbm),
        "hbm_peak_pct": max(hbm) if hbm else None,
        "hbm_peak_gb": (max(hbm) / 100 * 32) if hbm else None,
        "eval_summary": summary,
    }
    out["by_db"] = {
        db: {"n": len(rs), "correct": sum(1 for r in rs if r.get("correct"))}
        for db, rs in sorted(
            {db: [r for r in rows if r.get("db_id") == db] for db in {r.get("db_id") for r in rows}}.items()
        )
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "ex", "valid", "infra", "no_answer",
                                         "invalid_action_episodes", "lookup_rate",
                                         "hbm_peak_pct")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
