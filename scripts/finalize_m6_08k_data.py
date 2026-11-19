"""Export review/provenance evidence after data-only selection; never train.

Run with the project interpreter and the same PYTHONPATH as run_m6_08k_data.sh.
This exports evidence only: it cannot create a ready-for-training manifest.
"""
from collections import Counter
import json
from pathlib import Path

from e1_alignment.common import read_jsonl, write_json, write_jsonl, canonical_hash, sha256_file
from e1_alignment.revision.provenance import resolve
from e1_alignment.revision.selection import distribution

ART = Path("artifacts/m6/sft_data_revision")
REP = Path("reports/m6/sft_data_revision")


def main():
    rows = [r for name in ("audited_originals.jsonl", "audited_new.jsonl", "audited_model.jsonl")
            for r in read_jsonl(ART / name)]
    reviews = read_jsonl(REP / "individual_reviews.jsonl")
    reservations = read_jsonl(ART / "acquisition_ledger.jsonl")
    attempts = read_jsonl(ART / "gold_attempts.jsonl")
    by_review = {r["trajectory_id"]: r for r in reviews}
    diffs = []
    for r in read_jsonl(ART / "revisions.jsonl"):
        parent = resolve(r["origin"])
        boundary = r["revision_boundary"]
        diffs.append({"trajectory_id": r["trajectory_id"], "origin": r["origin"],
                      "parent_version_id": r["parent_version_id"], "boundary_action_count": boundary,
                      "original_complete_record": parent,
                      "replaced_suffix": parent["segments"][1 + 2 * boundary:],
                      "preserved_failure_prefix": r["segments"][:1 + 2 * boundary] if boundary else [],
                      "new_suffix": r["segments"][1 + 2 * boundary:],
                      "real_tool_receipt": r["acquisition_receipt"]})
    write_jsonl(ART / "revision_diffs.jsonl", diffs)
    write_jsonl(REP / "original_record_index.jsonl", [
        {"trajectory_id": r["trajectory_id"], "id": r["id"], "origin": r["origin"],
         "history_origin": r["history_origin"], "content_sha256": r["content_sha256"]} for r in rows])
    write_jsonl(REP / "rejected_and_pending.jsonl", [
        {"id": r["id"], "trajectory_id": r["trajectory_id"], "program_reasons": sorted(set(r["program_check"]["reasons"])),
         "review_status": by_review.get(r["trajectory_id"], {}).get("status", "pending"),
         "review": by_review.get(r["trajectory_id"])}
        for r in rows if not r["program_check"]["ok"] or by_review.get(r["trajectory_id"], {}).get("status") != "pass"])
    completed_gold = {r["reservation"]["identity"] for r in attempts}
    interrupted = [r for r in reservations if r["kind"] == "gold" and r["identity"] not in completed_gold]
    model_raw = read_jsonl(ART / "source_model.raw.jsonl")
    model_candidates = read_jsonl(ART / "model_candidates.jsonl")
    raw_by_state = Counter(r["collection_reservation"]["identity"] for r in model_raw)
    candidates_by_state = Counter(r["collection_reservation"]["identity"] for r in model_candidates)
    model_reservations = [r for r in reservations if r["kind"] == "model"]
    incomplete_model = [{**r, "saved_raw_records": raw_by_state[r["identity"]],
                         "saved_candidates": candidates_by_state[r["identity"]],
                         "status": "incomplete_or_interrupted_no_retry", "quota_consumed": True}
                        for r in model_reservations
                        if raw_by_state[r["identity"]] < r["reserved_candidates"] or
                        candidates_by_state[r["identity"]] < r["reserved_candidates"]]
    write_jsonl(REP / "interrupted_reservations.jsonl", interrupted + incomplete_model)
    reason_counts = Counter(reason for r in rows for reason in set(r["program_check"]["reasons"]))
    accounting = {"measurement": "实测", "candidate_records": len(rows),
                  "program_passed": sum(r["program_check"]["ok"] for r in rows),
                  "program_rejection_reasons_records_overlapping": dict(reason_counts),
                  "individual_reviews": dict(Counter(r["status"] for r in reviews)),
                  "quarantined_questions": sorted({r["id"] for r in reviews if r.get("quarantine_question")}),
                  "gold_reserved_attempts": sum(r["kind"] == "gold" for r in reservations),
                  "gold_completed_attempts": len(attempts), "gold_interrupted_attempts": len(interrupted),
                  "model_reserved_states": sum(r["kind"] == "model" for r in reservations),
                  "model_reserved_candidates": sum(r["reserved_candidates"] for r in reservations if r["kind"] == "model"),
                  "model_saved_raw_records": len(model_raw), "model_saved_candidates": len(model_candidates),
                  "model_complete_states": len(model_reservations) - len(incomplete_model),
                  "model_incomplete_states_quota_consumed": len(incomplete_model),
                  "gold_construction_measured_wall_s": sum(r["elapsed_s"] for r in attempts),
                  "individual_review_measured_wall_s": sum(r["measured_wall_s"] for r in read_jsonl(REP / "review_sessions.jsonl")),
                  "phase_wall_times": read_jsonl(REP / "phase_wall_times.jsonl"),
                  "model_trajectory_latency_ms": [r.get("latency_ms") for r in model_raw],
                  "sampling_run_checkpoints": [{"path": str(p), "report": json.loads(p.read_text())}
                                               for p in sorted((REP / "sampling_history").glob("*/model_collection.json"))] +
                                              ([{"path": str(REP / "model_collection.json"),
                                                 "report": json.loads((REP / "model_collection.json").read_text())}]
                                               if (REP / "model_collection.json").exists() else []),
                  "timing_scope": "Gold timing sums completed attempt wall times; interrupted attempt duration is unmeasured. Review wall is separate. Sampling phase wall includes startup, generation, tools and scoring; interrupted-run checkpoints are partial elapsed times, not full job totals. Per-trajectory latency includes tools and may overlap across candidates, so it must not be summed as pure inference time. Separate historical tool-only and generation-only latency was not instrumented. Audit resume elapsed excludes pre-resume segment and interpreter startup; see phase logs for measured scopes.",
                  "program_pool_distribution": distribution([r for r in rows if r["program_check"]["ok"]]),
                  "training_started": False, "capability_evaluation_started": False, "rl_started": False}
    write_json(REP / "delivery_accounting.json", accounting)
    evidence = [ART / "data_policy.json", REP / "preflight.json", REP / "individual_reviews.jsonl",
                ART / "revision_diffs.jsonl", REP / "candidate_ledger.jsonl", REP / "delivery_accounting.json"]
    evidence += [p for p in (ART / "acquisition_ledger.jsonl", ART / "gold_attempts.jsonl", ART / "source_model.raw.jsonl") if p.is_file()]
    evidence += [p for p in (REP / "lookup_capacity_certificate.json", REP / "review_corrections.jsonl",
                             REP / "tests_receipt_guard.log", REP / "receipt_guard_compatibility.json",
                             Path("scripts/certify_m6_08k_capacity.py"), Path(__file__)) if p.is_file()]
    write_json(REP / "evidence_hashes.json", {str(p): sha256_file(p) for p in evidence if p.is_file()})
    assessment = json.loads((REP / "assessment.json").read_text())
    status = assessment["status"]
    if status != "ready_for_sft_data" and (ART / "frozen_manifest.json").exists():
        raise ValueError("failed_delivery_must_not_have_frozen_manifest")
    lines = ["# M6-08k data-only 交付", "", f"状态：`{status}`。", "",
             "工程：程序通过、来源收据、历史保留、逐条审阅与可加载状态分别验收；程序通过不等于审阅批准。",
             f"实测候选 {accounting['candidate_records']}、程序通过 {accounting['program_passed']}，逐条审阅 {json.dumps(accounting['individual_reviews'], ensure_ascii=False)}，见 `delivery_accounting.json`。",
             f"Gold 构造实测预留 {accounting['gold_reserved_attempts']} 次，其中完成 {len(attempts)}、中断 {len(interrupted)}；所有预留均消耗预算。", "",
             f"源模型补采实测预留 {accounting['model_reserved_states']} 个状态，完整保存 {accounting['model_complete_states']} 个状态／{accounting['model_saved_candidates']} 条候选；{accounting['model_incomplete_states_quota_consumed']} 个未完整保存的状态仍消耗配额。",
             "学习信号：分布与不足见 `assessment.json`、`program_only_capacity.json`；Gold-assisted 与自主记录分开，未审和疑点记录不准入集。",
             "初始 lookup 的组合上界证明见 `lookup_capacity_certificate.json`；该证明把未审、可修复及剔除记录也纳入宽松上界，不授予任何逐条批准。",
             "任务能力：本轮未训练、未做新能力评测或 RL，不据数据通过率声称 EX 提升。", "",
             "来源索引、修订前后完整差异与失败前缀、真实工具收据、剔除待核验原因、审阅与独立计时均已留存。",
             "计时范围与未测部分明确写在 `delivery_accounting.json`，不将缺失时长当作零。"]
    (REP / "assessment.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "accounting": str(REP / "delivery_accounting.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
