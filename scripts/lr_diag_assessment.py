"""Assess the 3x5-step LR diagnostic: does the policy actually move?"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)")
METRICS = ("training/global_step", "critic/score/mean", "actor/kl_loss", "actor/pg_clipfrac",
           "actor/entropy", "actor/grad_norm", "response_length/mean", "num_turns/mean",
           "timing_s/step", "actor/lr")


def parse_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if "step:" not in line or "training/global_step" not in line:
            continue
        kv = {k: float(v) for k, v in PAIR.findall(line)}
        if "training/global_step" in kv:
            rows.append({k: kv.get(k) for k in METRICS})
    return sorted(rows, key=lambda r: r["training/global_step"])


def parse_traces(path):
    rows = []
    if not path.is_dir():
        return rows
    for file in sorted(path.glob("rollouts.worker-*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trace_stats(rows):
    by_step = defaultdict(list)
    for row in rows:
        by_step[row.get("step")].append(row)
    out = {}
    for step, items in sorted(by_step.items()):
        first = Counter()
        seq = Counter()
        no_tools = 0
        lookups = 0
        sqls = 0
        finish = Counter()
        for row in items:
            events = [e for e in row.get("tool_events", []) if not e.get("seeded")]
            acts = tuple(e["action"] for e in events)
            first[acts[0] if acts else "none"] += 1
            seq[acts] += 1
            no_tools += 0 if acts else 1
            lookups += row.get("schema_lookups", 0)
            sqls += row.get("sql_executes", 0)
            finish[row.get("finish_reason")] += 1
        n = len(items)
        out[step] = {
            "n": n,
            "first_action": dict(first),
            "lookup_first_rate": first.get("schema_lookup", 0) / max(n, 1),
            "zero_tool_rate": no_tools / max(n, 1),
            "avg_schema_lookups": lookups / max(n, 1),
            "avg_sql_executes": sqls / max(n, 1),
            "recovery_template_rate": seq[("sql_execute", "schema_lookup", "sql_execute")] / max(n, 1),
            "finish_reasons": dict(finish),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["lr1e6", "lr2e6", "lr5e6"])
    p.add_argument("--out", default="reports/m6/lr_diag_assessment.json")
    a = p.parse_args()

    report = {"arms": {}}
    for tag in a.tags:
        log = Path("logs") / f"lr_diag_{tag}.log"
        trace = Path("reports/m6/traces_lr_diag") / tag
        arm = {"log": str(log), "trace": str(trace)}
        if log.is_file():
            arm["steps"] = parse_log(log)
        arm["behavior"] = trace_stats(parse_traces(trace))
        report["arms"][tag] = arm
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for tag, arm in report["arms"].items():
        print(f"=== {tag} ===")
        for step in arm.get("steps", []):
            print(" step", int(step["training/global_step"]),
                  "score", round(step["critic/score/mean"], 4) if step["critic/score/mean"] is not None else None,
                  "kl", step["actor/kl_loss"], "clip", step["actor/pg_clipfrac"],
                  "ent", round(step["actor/entropy"], 4) if step["actor/entropy"] else None,
                  "grad", round(step["actor/grad_norm"], 4) if step["actor/grad_norm"] else None,
                  "turns", round(step["num_turns/mean"], 3) if step["num_turns/mean"] else None)
        for step, b in arm["behavior"].items():
            print(" trace step", step, "n", b["n"], "first", b["first_action"],
                  "lookup_first", round(b["lookup_first_rate"], 3),
                  "zero_tool", round(b["zero_tool_rate"], 3),
                  "recovery", round(b["recovery_template_rate"], 3))


if __name__ == "__main__":
    main()
