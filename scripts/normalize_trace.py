"""Fill in the missing `step` field of a training trace.

The veRL patch that injects step/sample/rollout into the agent loop proved fragile
(a reset of verl/experimental/agent_loop/agent_loop.py silently removed it), so the
analysis must not depend on it. Rollout records are appended per worker in step
order, and every worker handles the same number of prompts per step, so a fixed
chunk of records per worker recovers the step index.

Usage: python3 scripts/normalize_trace.py --trace reports/m6/traces_stage1 \
           --per-worker-step 8 --out reports/m6/traces_stage1/rollouts.normalized.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--per-worker-step", type=int, required=True,
                   help="records per worker per optimizer step (= n * prompts_per_worker)")
    p.add_argument("--out", default="")
    a = p.parse_args()

    trace = Path(a.trace)
    files = sorted(trace.glob("rollouts*.jsonl")) if trace.is_dir() else [trace]
    out_path = Path(a.out) if a.out else trace / "rollouts.normalized.jsonl"
    rows = []
    for file in files:
        if file.name == "rollouts.normalized.jsonl":
            continue
        records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, record in enumerate(records):
            record.setdefault("worker_file", file.name)
            if record.get("step") is None:
                record["step"] = index // a.per_worker_step + 1
            rows.append(record)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in rows:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    steps = sorted({r["step"] for r in rows})
    print(json.dumps({"records": len(rows), "files": len(files), "steps": len(steps),
                      "step_range": [steps[0], steps[-1]] if steps else [], "out": str(out_path)}))


if __name__ == "__main__":
    main()
