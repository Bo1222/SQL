"""Sequential acquisition/audit driver; never substitutes machine reviews.

Can wait for an already launched acquisition. Stops at a proven structural data
failure, or hands an explicit pending-review queue back to the assistant.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from e1_alignment.common import read_jsonl, write_json, write_jsonl
from e1_alignment.rebuild.freeze import quota_feasibility

ART = Path("artifacts/m6/sft_rebuild")
REP = Path("reports/m6/sft_rebuild")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait-for-existing-b", action="store_true")
    a = p.parse_args()
    if a.wait_for_existing_b:
        while True:
            state = json.loads((REP / "npu_budget.json").read_text())
            jobs = [j for j in state["jobs"] if j["stage"] == "B"]
            if jobs and jobs[-1]["status"] != "running":
                if jobs[-1]["status"] != "complete":
                    raise SystemExit("existing B acquisition did not complete; inspect journal")
                break
            time.sleep(5)
    # B is the binding quota: A/C acquisition cannot increase the B count.
    # Audit the entire available B pool before spending on unrelated deficits.
    with (REP / "audit_b_pool.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "e1_alignment.rebuild.audit", "--candidates",
                        "artifacts/m6/e1_alignment/e1_self_rollouts.jsonl",
                        "artifacts/m6/e1_alignment/e1_repair_rollouts.jsonl",
                        "artifacts/m6/e1_alignment/e1_gold_assisted.jsonl", str(ART / "new_B.jsonl"),
                        "--prefixes", "artifacts/m6/e1_alignment/e1_failure_prefixes.jsonl", str(ART / "new_B_prefixes.jsonl"),
                        "--out", str(ART / "quota_preflight_candidates.jsonl"), "--ledger", str(REP / "quota_preflight_quality_ledger.jsonl"),
                        "--summary", str(REP / "quota_preflight_audit_summary.json")], stdout=log, stderr=subprocess.STDOUT, check=True)
    from collections import defaultdict
    from e1_alignment.rebuild.freeze import learning_value
    b_good = [r for r in read_jsonl(ART / "quota_preflight_candidates.jsonl") if r["program_check"]["ok"] and r["trajectory_source"] == "e1_repair_continuation"]
    per_question = defaultdict(set)
    for row in b_good:
        per_question[row["id"]].add(learning_value(row))
    upper = sum(min(2, len(values)) for values in per_question.values())
    certificate = {"measurement": "实测候选与程序检查，组合上界（非任务能力）", "b_program_passed": len(b_good),
                   "b_questions": len(per_question), "b_max_after_question_and_learning_value_caps": upper,
                   "minimum_dataset": 400, "b_ratio_minimum": .25, "minimum_required_b": 100,
                   "semantic_approved": 0, "training_started": False,
                   "decision": "stop_at_data_gate" if upper < 100 else "continue_acquisition",
                   "reason": "A/C cannot fill the B quota; no quota relaxation or duplicate filling",
                   "unused_acquisition": ["A", "C"] if upper < 100 else [],
                   "scope": "program-only structural upper bound, NOT semantic approval"}
    write_json(REP / "b_quota_certificate.json", certificate)
    if upper < 100:
        write_json(REP / "program_only_feasibility.json", certificate)
        print(json.dumps(certificate), flush=True)
        return
    for source, seconds, count in (("A", 3300, 800), ("C", 600, 200)):
        if (REP / f"collection_{source}.json").exists():
            summary = json.loads((REP / f"collection_{source}.json").read_text())
            if summary.get("complete"):
                continue
            raise SystemExit(f"partial {source} journal requires audit before any additional sampling")
        with (REP / f"collection_{source}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "e1_alignment.rebuild.budget", "--stage", source,
                            "--seconds", str(seconds), "--count", str(count), "--", "bash",
                            "scripts/run_m6_08j_collect.sh", source], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (REP / "audit_all.log").open("w") as log:
        command = [sys.executable, "-m", "e1_alignment.rebuild.audit", "--candidates"]
        command += [str(Path("artifacts/m6/e1_alignment") / f) for f in
                    ("e1_self_rollouts.jsonl", "e1_repair_rollouts.jsonl", "e1_gold_assisted.jsonl")]
        command += [str(ART / f"new_{s}.jsonl") for s in "ABC"]
        command += ["--prefixes", "artifacts/m6/e1_alignment/e1_failure_prefixes.jsonl",
                    str(ART / "new_B_prefixes.jsonl"), "--out", str(ART / "all_audited_candidates.jsonl"),
                    "--ledger", str(REP / "all_quality_ledger.jsonl"), "--summary", str(REP / "all_audit_summary.json")]
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    all_rows = read_jsonl(ART / "all_audited_candidates.jsonl")
    passing = [r for r in all_rows if r["program_check"]["ok"]]
    picked, feasibility = quota_feasibility(passing)
    feasibility["scope"] = "optimistic program-only upper bound, NOT semantic approval or a frozen dataset"
    feasibility["semantic_approved"] = 0
    feasibility["training_started"] = False
    feasibility["decision"] = "pending_individual_review" if picked else "stop_at_data_gate"
    write_json(REP / "program_only_feasibility.json", feasibility)
    if picked:
        # Proposed reading queue is not loadable by the trainer.
        write_jsonl(ART / "individual_review_queue.jsonl", picked)
    print(json.dumps(feasibility), flush=True)


if __name__ == "__main__":
    main()
