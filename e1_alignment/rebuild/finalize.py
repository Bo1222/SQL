"""Materialize the registered data-stage stop, never fabricate a trained candidate."""
from collections import Counter
import json
from pathlib import Path

from e1_alignment.common import read_jsonl, write_json, write_jsonl, sha256_file
from e1_alignment.rebuild import RULE_VERSION
from e1_alignment.rebuild.freeze import distribution
from e1_alignment.rebuild.quality import review_errors

ART = Path("artifacts/m6/sft_rebuild")
REP = Path("reports/m6/sft_rebuild")


def main():
    certificate = json.loads((REP / "b_quota_certificate.json").read_text())
    collection = json.loads((REP / "collection_B.json").read_text())
    audit = json.loads((REP / "quota_preflight_audit_summary.json").read_text())
    budget = json.loads((REP / "npu_budget.json").read_text())
    contract = json.loads((REP / "source_contract_verification.json").read_text())
    tests = json.loads((REP / "test_status.json").read_text())
    if certificate["decision"] != "stop_at_data_gate" or not collection.get("complete"):
        raise SystemExit("not a completed registered data stop")
    if collection["attempts"] != 600 or len(read_jsonl(ART / "new_B.raw.jsonl")) != 600:
        raise SystemExit("B sampling budget not fully accounted")
    if any(j["status"] == "running" for j in budget["jobs"]):
        raise SystemExit("job still running")
    if (ART / "train.jsonl").exists() or Path("outputs/sft_rebuild/adapter").exists():
        raise SystemExit("unexpected training artifacts at a data stop")
    if not contract["all_frozen_runtime_model_tokenizer_database_files_match"]:
        raise SystemExit("source runtime contract mismatch")
    final_candidates = Path(certificate.get("final_candidates", ART / "quota_preflight_candidates.jsonl"))
    rows = read_jsonl(final_candidates)
    audit["semantic_approved_by_program"] = audit.pop("semantic_approved", 0)
    audit["initial_program_passed"] = audit["program_passed"]
    audit["program_passed"] = sum(r["program_check"]["ok"] for r in rows)
    audit["rejections"] = dict(Counter(reason for r in rows for reason in r["program_check"]["reasons"]))
    audit["program_pass_distribution"] = distribution([r for r in rows if r["program_check"]["ok"]])
    audit["final_candidates"] = str(final_candidates)
    audit["failure_prefix_final_check"] = "reports/m6/sft_rebuild/failure_prefix_final_check.json"
    write_json(REP / "final_audit_summary.json", audit)
    reviews = {r["trajectory_id"]: r for r in read_jsonl(REP / "individual_reviews.jsonl")}
    findings = {r["trajectory_id"]: r for r in read_jsonl(REP / "individual_review_findings.initial.jsonl")}
    ledger, approved = [], []
    for r in rows:
        review = reviews.get(r["trajectory_id"])
        errors = review_errors(r, review)
        status = "excluded" if not r["program_check"]["ok"] else ("pass" if not errors else "pending")
        if status == "pass":
            approved.append(r)
        ledger.append({"trajectory_id": r["trajectory_id"], "id": r["id"], "db_id": r["db_id"],
                       "source": r["trajectory_source"], "content_sha256": r["content_sha256"],
                       "origin": r["origin"], "parent_version_id": r["parent_version_id"],
                       "status": status, "program_passed": r["program_check"]["ok"],
                       "program_reasons": r["program_check"]["reasons"], "review_errors": errors,
                       "semantic_review": review,
                       "earlier_pending_finding": findings.get(r["trajectory_id"]),
                       "included_in_training": False})
    write_jsonl(REP / "quality_ledger.final.jsonl", ledger)
    bad = []
    for index, r in enumerate(read_jsonl(ART / "new_B.raw.jsonl"), 1):
        if not r.get("final_correct") or r.get("excluded_reason"):
            bad.append({"id": r["id"], "origin_line": index, "state_id": r.get("failure_state", {}).get("state_id"),
                        "reason": r.get("excluded_reason") or "final_ex_zero", "source": str(ART / "new_B.raw.jsonl"),
                        "included_in_training": False})
    write_jsonl(REP / "new_b_badcase_index.jsonl", bad)
    write_jsonl(REP / "record_revisions.jsonl", [])
    passed = [r for r in rows if r["program_check"]["ok"]]
    measured_review = json.loads((REP / "review_sessions.json").read_text())
    total_wall = sum(j.get("charged_s", j["reserved_s"]) for j in budget["jobs"])
    if total_wall > budget["limit_s"]:
        raise SystemExit("total job budget exceeded")
    paths = [final_candidates, ART / "quota_preflight_candidates.jsonl", ART / "new_B.raw.jsonl", ART / "new_B.jsonl",
             ART / "new_B_prefixes.jsonl", ART / "monitor64.jsonl", REP / "quality_ledger.final.jsonl",
             REP / "individual_reviews.jsonl", REP / "individual_review_findings.initial.jsonl"]
    manifest = {"status": "rejected_at_data_gate", "frozen_training_dataset": None, "rule_version": RULE_VERSION,
                "candidate_distribution": distribution(rows), "program_distribution": distribution(passed),
                "individually_approved_distribution": distribution(approved), "included_records": 0,
                "b_quota_certificate": certificate, "evidence_hashes": {str(p): sha256_file(p) for p in paths},
                "note": "Candidate evidence hashes are not a training manifest. No unreviewed row is loadable."}
    write_json(REP / "rejected_dataset_manifest.json", manifest)
    report = {"task": "M6-08j", "status": "stopped_at_registered_data_gate", "measurement": "实测及明确标注的组合上界",
              "collection_B": collection, "audit": audit, "quota_certificate": certificate,
              "quality_status_counts": dict(Counter(r["status"] for r in ledger)),
              "individually_approved": len(approved), "fully_reviewed_training_dataset": False,
              "included_records": 0, "record_revisions_submitted": 0,
              "measured_individual_review_wall_s": sum(s["seconds"] for s in measured_review["sessions"]),
              "exploratory_review_time": measured_review["earlier_exploratory_findings_time"],
              "npu_job_wall_s": total_wall, "npu_budget_s": budget["limit_s"],
              "source_contract": contract, "tests": tests,
              "stages": {"data": "failed_quota", "A_supplement": "not_run_cannot_fill_B_quota",
                         "C_supplement": "not_run_cannot_fill_B_quota", "training": "not_run",
                         "monitor": "not_run", "new_dev200": "not_run", "fixed_prefix_recovery": "not_run", "RL": "paused"},
              "conclusions": {"engineering": "Quality gates and corrected training math implemented and tested; formal data admission failed.",
                              "learning_signal": "Even optimistic program-passing B pool cannot meet the registered minimum; no dataset frozen.",
                              "task_capability": "No candidate trained or evaluated; no new capability conclusion."}}
    write_json(REP / "assessment.json", report)
    upper, needed = certificate["b_max_after_question_and_learning_value_caps"], certificate["minimum_required_b"]
    text = f'''# M6-08j 执行结果：停在数据门槛

**负结果：本轮不训练。** B 补采预算已用满，实测 {collection['attempts']} 候选 / {collection['positive']} 最终 EX 成功；成功不等于合格。合并旧 B 后，程序检查通过 {certificate['b_program_passed']} 条；按同题上限及不同学习价值规则，由实测池计算的乐观入集上界只有 **{upper} 条 B**，低于方案要求的至少 **{needed} 条 B**（最少数据集 {certificate['minimum_dataset']} × B 比例 {certificate['b_ratio_minimum']:.0%}）。来源 `collection_B.json`、`b_quota_certificate.json`、`final_audit_summary.json`。上界不是任务能力实测，也不是收益外推。

[已规划 M6-08j] 按停止规则结束本轮数据阶段。A/C 不能增加 B 数量，因此未消耗其补采额度；未放宽配额、未复制凑数、未追加模型采样。未冻结训练集，未训练候选，未运行新的监控/dev200/固定前缀恢复评测。RL 保持暂停。保留旧 V2 失败对照。

## 工程

实现固定 SQLGlot、真实 schema/作用域绑定、实际可见且在动作前的覆盖证据、JOIN 语义审阅入口、原始 Gold/每个受监督 execute 的真实完整执行评分、真实失败日志核验、逐段 mask、修订收据、审阅/内容/规则哈希、训练时独立数据隔离复核、完整累计窗口 token 平均、单候选/超时/总预算约束及固定分母评测入口。当前测试：{tests['result']}，来源 `{tests['log']}`；测试验证数学与入口，未发生新 NPU 训练。

原训练器在 `archive/m6_08i/sft_train.py` 逐字节保留，SHA 与旧 fingerprint 一致（`legacy_trainer_preservation.json`）。冻结的运行时/模型/tokenizer/DB 文件实测一致（`source_contract_verification.json`）；新 pipeline 与审计代码独立记录。

## 学习信号

整池审计共 {len(rows)} 条既有/新增正候选，实测程序通过 {len(passed)} 条（来源 `final_audit_summary.json`）。逐条审阅实测批准 {len(approved)} 条；完整训练集逐条审阅未完成，不能宣称所有候选语义合格。结构数量关卡先失败，没有拟放行的最终数据集。已审阅的真实县域邮编修复记录具备问题/Gold/SQL一致性、可见 schema 和独立 FK/结果核验；另有用户数与交易行计数的疑点仍待核验。只有实际逐条阅读才产生审阅结论，程序检查不会批量批准语义。审阅记录/时间见 `individual_reviews.jsonl`、`review_sessions.json`；最终质量账本见 `quality_ledger.final.jsonl`。

失败来源元数据存在不一致的记录被保留并剔除，未手工改写历史；即使恢复这些仅来源核验失败的候选，数量也不能达到方案门槛。B 的最终失败核验还排除了只有 lookup、没有失败 SQL 的前缀，不能把检索后的答题成功冒充失败恢复（`failure_prefix_final_check.json`）。旧 C 的机械表名首查不符合本轮问题/evidence 首查策略；其被排除是本轮数据规则的结果，不是对 V2 退化原因的因果判定。没有修改题目、Gold 或工具 Observation。没有提交轨迹修订版本，相关原始记录与剔除原因均保留。

## 任务能力

无新候选、无新能力评测，不能认定重建方案提高或降低 EX。历史 V2 的完整负结果仍有效；`historical_metric_denominator_check.json` 只重算旧完整记录：首候选包含 execute/直接 answer，固定全题数分母；不是新评测或新收益。没有以训练 loss、lookup 或 valid 代替任务能力。

## 交付与预算

`rejected_dataset_manifest.json` 保存候选、账本、监控集及其哈希，明确不是可训练 manifest。初始动作/来源/错误类型/监督 token 分布保存在该文件。新 B 原始采样及未成功续接均保留，`new_b_badcase_index.jsonl` 可定位；总作业墙钟实测 {total_wall:.3f} s（`npu_budget.json`），未超过方案总预算 {budget['limit_s']} s。单独计时的逐条审阅实测 {report['measured_individual_review_wall_s']:.3f} s（`review_sessions.json`）；更早探索阅读未计时，明确未纳入该实测值。源 `step_600_merged` 和旧基线/失败分支未覆盖。
'''
    (REP / "assessment.md").write_text(text)
    # Update only the registered task and current-state anchor, retaining old evidence.
    progress = Path("docs/PROGRESS.md").read_text()
    start = progress.index("**[已规划 M6-08j]")
    end = progress.index("\n\n**RL 暂停", start)
    anchor = (f"**[已规划 M6-08j] 已按数据关卡停止：B 实测 {collection['attempts']} 候选 / {collection['positive']} 最终 EX 成功；"
              f"合并旧 B 后程序通过 {certificate['b_program_passed']}，按实测池计算的选择上界 {upper} < 方案门槛 {needed}。"
              f"未冻结训练集、未训练/评测新候选；RL 暂停。来源 `reports/m6/sft_rebuild/assessment.json`、`b_quota_certificate.json`；"
              f"测试 {tests['result']}（`{tests['log']}`）。**")
    progress = progress[:start] + anchor + progress[end:]
    progress = progress.replace("[已规划 M6-08j] 用户已授权本次重建及一个候选；按质量关卡、预算和停止判据执行，详见 `docs/M6_08j_REBUILD.md`。不得自动追加候选、epoch 或恢复 RL。",
                                "[已规划 M6-08j] 已执行并按 B 数据数量关卡停止；详见 `reports/m6/sft_rebuild/assessment.md`。本轮未产生新候选，不追加采样、训练或恢复 RL。")
    progress = progress.replace("M6-08j 现已单独授权，先验证数据关卡，未通过不训练。", "M6-08j 已在数据数量关卡停止，未训练。")
    lines = progress.splitlines()
    if not any(line.startswith("| **M6-08j") for line in lines):
        idx = next(i for i, line in enumerate(lines) if line.startswith("| **M6-08i /"))
        lines.insert(idx + 1, f"| **M6-08j 数据质量重建** | **❌ 数据门槛未过，已停止** | `reports/m6/sft_rebuild/assessment.json`、`b_quota_certificate.json` | B 选择上界 {upper} < 方案门槛 {needed}；未训练/未新增评测，RL 暂停 |")
    Path("docs/PROGRESS.md").write_text("\n".join(lines) + "\n")
    tasks = Path("docs/TASKS_M6.md").read_text()
    start, end = tasks.index("### P2 · M6-08j"), tasks.index("### P2 · M6-09")
    section = tasks[start:end].replace("**状态：进行中（用户已授权执行本方案，2026-11-10）。**",
                                    "**状态：❌ 已执行，按数据门槛停止（2026-11-10），未训练。**")
    section += (f"**执行结果（实测与组合上界分开）**：B {collection['attempts']} 候选 / {collection['positive']} 最终 EX 成功；"
                f"合并旧 B 后程序通过 {certificate['b_program_passed']}，选择组合上界 {upper}，低于设计门槛 {needed}。"
                f"未冻结训练集；新训练、监控、dev200、固定前缀恢复未执行；RL 暂停。测试 {tests['result']}。"
                "证据：`reports/m6/sft_rebuild/assessment.md`、`b_quota_certificate.json`、`test_status.json`。\n\n")
    Path("docs/TASKS_M6.md").write_text(tasks[:start] + section + tasks[end:])
    print(json.dumps({"stopped": True, "b_upper_bound": upper, "b_required": needed,
                      "individually_approved": len(approved), "training_started": False}))


if __name__ == "__main__":
    main()
