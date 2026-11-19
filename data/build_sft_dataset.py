"""Normalize the frozen multi-turn SFT trajectories into the plan's data contract.

Input rows already carry the full trajectory; this step only reshapes fields,
derives the supervision mask and classifies trajectory provenance.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MESSAGES_ROLE = {"prompt": "user", "action": "assistant", "observation": "tool"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(row, metadata_version="v1"):
    trajectory = row["trajectory"]
    segments = trajectory.get("segments", [])
    messages = [{"role": MESSAGES_ROLE.get(s["kind"], "user"), "content": s["text"]} for s in segments]
    response_mask = [1 if s["kind"] == "action" else 0 for s in segments]
    first_sql, final_sql = trajectory.get("first_sql"), trajectory.get("final_sql")
    if final_sql is None:
        source_type = "invalid"
    elif first_sql and final_sql and first_sql.strip() != final_sql.strip():
        source_type = "autonomous_recovery"
    else:
        source_type = "autonomous_success"
    errors = []
    for execution in trajectory.get("executions", []):
        error = (execution.get("result") or {}).get("error_type")
        if error:
            errors.append(error)
    return {
        "question_id": row["id"],
        "db_id": row["db_id"],
        "db_path": row.get("db_path"),
        "messages": messages,
        "trajectory": {"segments": segments, "tool_calls": trajectory.get("tool_calls"),
                       "actions": trajectory.get("actions"),
                       "stop_reason": trajectory.get("stop_reason")},
        "final_sql": final_sql,
        "final_ex": bool(row.get("final", {}).get("correct")),
        "source_type": source_type,
        "tool_calls": trajectory.get("tool_calls", 0),
        "error_types": errors,
        "response_mask": response_mask,
        "metadata_version": metadata_version,
        "trajectory_id": hashlib.sha256(
            json.dumps([row["id"], final_sql], ensure_ascii=False).encode()).hexdigest()[:16],
        "supervision_policy": row.get("supervision_policy"),
        "token_count": sum(len(s.get("ids", [])) for s in segments),
        "masked_tokens": sum(len(s.get("ids", [])) for s in segments if s["kind"] == "action"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/bird_agent/full_sft_20261026/full_tool.jsonl")
    p.add_argument("--out", default="artifacts/datasets/sft_trajectories.jsonl")
    p.add_argument("--summary", default="artifacts/manifests/sft_dataset_summary.json")
    a = p.parse_args()

    rows = read_jsonl(a.data)
    records = [normalize(row) for row in rows]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    summary = {
        "n": len(records),
        "dbs": len({r["db_id"] for r in records}),
        "source_types": dict(Counter(r["source_type"] for r in records)),
        "final_ex_rate": sum(1 for r in records if r["final_ex"]) / max(len(records), 1),
        "avg_tool_calls": sum(r["tool_calls"] for r in records) / max(len(records), 1),
        "avg_masked_tokens": sum(r["masked_tokens"] for r in records) / max(len(records), 1),
        "error_type_counts": dict(Counter(e for r in records for e in r["error_types"])),
        "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "records": str(out),
    }
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
