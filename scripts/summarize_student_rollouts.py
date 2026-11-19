"""Compare M6-08m P1/P2 rollout arms.

Usage:

    python scripts/summarize_student_rollouts.py \
      --arm e1=reports/m6/e1_student_only/p1_e1 \
      --arm e1_1=reports/m6/e1_student_only/p1_e1_1 \
      --out-dir reports/m6/e1_student_only/p1_comparison

    python scripts/summarize_student_rollouts.py \
      --arm n1=reports/m6/e1_student_only/p2_n1 \
      --arm n4=reports/m6/e1_student_only/p2_n4 \
      --arm n8=reports/m6/e1_student_only/p2_n8 \
      --arm repair=reports/m6/e1_student_only/p2_repair_k4 \
      --out-dir reports/m6/e1_student_only/p2_comparison
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_sql(sql):
    return " ".join((sql or "").split()).lower().rstrip(";").strip()


def mcnemar_exact(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def load_arm(name, path):
    rows = list(read_jsonl(Path(path) / "rollouts.jsonl"))
    if not rows:
        raise SystemExit(f"{path}/rollouts.jsonl is empty")
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    success = {q for q, items in by_question.items() if any(r.get("final_ex") for r in items)}
    success_sql = {norm_sql(r["final_sql"]) for r in rows
                   if r.get("final_ex") and r.get("final_sql")}
    base_rows = [r for r in rows if int(r.get("repair_round") or 0) == 0]
    repair_rows = [r for r in rows if int(r.get("repair_round") or 0) > 0]
    base_success = {r["question_id"] for r in base_rows if r.get("final_ex")}
    repair_success = {r["question_id"] for r in repair_rows if r.get("final_ex")}
    return {
        "name": name,
        "path": str(path),
        "rows": rows,
        "by_question": by_question,
        "metrics": {
            "tasks": len(by_question),
            "rollouts": len(rows),
            "success_rollouts": sum(bool(r.get("final_ex")) for r in rows),
            "success_questions": len(success),
            "coverage": round(len(success) / max(len(by_question), 1), 4),
            "unique_success_sql": len(success_sql),
            "invalid_actions": sum(int((r.get("record") or {}).get("invalid_actions") or 0) for r in rows),
            "infrastructure_errors": sum(bool((r.get("record") or {}).get("infrastructure_error")) for r in rows),
            "exceptions": sum(bool(r.get("exception")) for r in rows),
            "lookup_empty": sum(int(r.get("lookup_empty") or 0) for r in rows),
            "lookup_error": sum(int(r.get("lookup_error") or 0) for r in rows),
            "sql_error": sum(int(r.get("sql_error") or 0) for r in rows),
            "wall_s": round(sum(float(r.get("wall_s") or 0) for r in rows), 3),
            "provenance_counts": dict(Counter((r.get("provenance") or {}).get("source") for r in rows)),
            "base_rollouts": len(base_rows),
            "repair_rollouts": len(repair_rows),
            "base_success_questions": len(base_success),
            "repair_new_success_questions": len(repair_success - base_success),
        },
        "success": success,
        "base_success": base_success,
        "repair_success": repair_success,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True,
                        help="name=out_dir; repeat for each arm")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    arms = []
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms.append(load_arm(name, path))
    baseline = arms[0]
    total_tasks = set(baseline["by_question"])
    for arm in arms[1:]:
        if set(arm["by_question"]) != total_tasks:
            raise SystemExit(f"task sets differ: {baseline['name']} vs {arm['name']}")

    comparisons = []
    for arm in arms[1:]:
        gain = len(arm["success"] - baseline["success"])
        loss = len(baseline["success"] - arm["success"])
        comparisons.append({
            "baseline": baseline["name"], "arm": arm["name"],
            "gain": gain, "loss": loss,
            "net": gain - loss,
            "delta_coverage": round(arm["metrics"]["coverage"] - baseline["metrics"]["coverage"], 4),
            "mcnemar_exact_p": round(mcnemar_exact(gain, loss), 6),
            "success_rollouts_delta": arm["metrics"]["success_rollouts"] - baseline["metrics"]["success_rollouts"],
            "rollouts_delta": arm["metrics"]["rollouts"] - baseline["metrics"]["rollouts"],
        })

    summary = {
        "baseline": baseline["name"],
        "arms": {arm["name"]: arm["metrics"] for arm in arms},
        "comparisons": comparisons,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = ["# M6-08m rollout comparison", "",
             f"- baseline: **{baseline['name']}**", "",
             "## Arms", "",
             "| arm | tasks | rollouts | success rollouts | success questions | coverage | "
             "unique success SQL | lookup empty | SQL error | wall s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        m = arm["metrics"]
        lines.append(f"| {arm['name']} | {m['tasks']} | {m['rollouts']} | {m['success_rollouts']} | "
                     f"{m['success_questions']} | {m['coverage']:.1%} | {m['unique_success_sql']} | "
                     f"{m['lookup_empty']} | {m['sql_error']} | {m['wall_s']:.1f} |")
    lines += ["", "## Paired comparisons vs baseline", "",
              "| arm | gain | loss | net | Δcoverage | McNemar p | Δsuccess rollouts | Δrollouts |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for comp in comparisons:
        lines.append(f"| {comp['arm']} | {comp['gain']} | {comp['loss']} | {comp['net']} | "
                     f"{comp['delta_coverage']:+.1%} | {comp['mcnemar_exact_p']:.4f} | "
                     f"{comp['success_rollouts_delta']:+d} | {comp['rollouts_delta']:+d} |")
    lines += ["", "## Repair detail", ""]
    for arm in arms:
        m = arm["metrics"]
        if m["repair_rollouts"]:
            lines.append(f"- {arm['name']}: base success questions **{m['base_success_questions']}**, "
                         f"repair rollouts **{m['repair_rollouts']}**, "
                         f"new success questions from repair **{m['repair_new_success_questions']}**")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
