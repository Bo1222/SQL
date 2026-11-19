"""Assemble the measured 3B/harness assessment from saved evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1_alignment.common import read_jsonl, sha256_file, write_json
from eval.paired_statistics import bootstrap_ci, mcnemar

OUT = ROOT / "reports/m6/harness_audit_20261117"


def load(path):
    return json.loads(Path(path).read_text())


def main():
    audits = {name: load(OUT / name / "summary.json") for name in (
        "dev200_e1_1", "test1534_e1_1", "dev200_e1_2", "test1534_e1_2")}
    probe = load(OUT / "oracle_probe50/summary.json")
    if not probe["completed"]:
        raise SystemExit("oracle probe is incomplete; do not finalize an EX comparison")
    scores = load(OUT / "scoring_contract/summary.json")
    boundaries = load(OUT / "boundaries.json")
    dev = audits["dev200_e1_1"]["datasets"]["dev200"]
    test = audits["test1534_e1_1"]["datasets"]["test1534"]
    fixed = audits["test1534_e1_2"]["datasets"]["test1534"]
    history = audits["test1534_e1_1"]["existing_traces"]["test1534_sft"]
    records = read_jsonl(OUT / "oracle_probe50/episodes.jsonl")
    old = {r["id"]: r for r in read_jsonl(ROOT / "reports/m6/e1_student_only/sft_lowlr_788.jsonl")}
    current = [r for r in records if r["arm"] == "current"]
    probe["historical_current_correct_agreement"] = sum(r["correct"] == old[r["id"]]["correct"] for r in current)
    probe["historical_current_final_sql_exact_agreement"] = sum(r["final_sql"] == old[r["id"]]["final_sql"] for r in current)
    probe["mcnemar_exact_p"] = mcnemar(probe["gain"], probe["loss"])
    probe["oracle_finish_reasons"] = dict(Counter(r["finish_reason"] for r in records if r["arm"] == "oracle_schema"))
    relabeling = {}
    for name, s in scores["sources"].items():
        total = s["historical_correct"] + s["set_only_additional_correct"]
        relabeling[name] = {"n": s["n"], "project_strict_correct": s["historical_correct"],
                           "set_only_additional_correct": s["set_only_additional_correct"],
                           "derived_set_only_correct": total,
                           "derived_set_only_ex": total / s["n"],
                           "kind": "posthoc_saved_prediction_relabeling_not_new_model_run"}
    changed = defaultdict(set)
    for item in scores["differences"]:
        if item["score"].get("bird_set_only_correct"):
            changed[item["trace"]].add(item["id"])
    base = {r["id"]: r for r in read_jsonl(ROOT / "reports/m6/e1_student_only/test1534_base_e1_1.jsonl")}
    sft = {r["id"]: r for r in read_jsonl(ROOT / "reports/m6/e1_student_only/test1534_sft_lowlr_788.jsonl")}
    gain = loss = 0
    by_db = defaultdict(list)
    for qid in base:
        ac = base[qid]["correct"] or qid in changed["test1534_base"]
        bc = sft[qid]["correct"] or qid in changed["test1534_sft"]
        gain += bool(bc and not ac)
        loss += bool(ac and not bc)
        by_db[base[qid]["db_id"]].append(int(bc) - int(ac))
    relabeled_pair = {"gain": gain, "loss": loss, "mcnemar_exact_p": mcnemar(gain, loss),
                      "db_bootstrap_ci95": bootstrap_ci(by_db),
                      "kind": "posthoc_saved_prediction_relabeling_not_new_model_run"}
    summary = {"date": "2026-11-17", "gold_submission_dev200": dev,
        "gold_submission_test1534": test, "fixed_lookup_search_test1534": fixed,
        "historical_test1534_sft_errors": history, "oracle_probe50": probe,
        "scoring_relabeling": relabeling, "relabeled_test1534_pair": relabeled_pair,
        "boundaries": boundaries,
        "code_changes": ["eval/sql_dependencies.py", "eval/gold_extraction.py", "eval/evaluate_agent.py",
                         "retrieval/e1_2.py", "tools/schema_lookup.py", "agent/sql_episode.py",
                         "tools/sql_worker.py", "tools/sql_execute.py", "reward/bird_ex_reward.py"],
        "tests": (OUT / "tests.log").read_text(),
        "limits": ["Gold-guided diagnostics are not deployable EX or a global capability ceiling.",
                   "No SFT/RL was run, no model weights changed, no training data was written.",
                   "E1.2/compact_v2/bird_set are opt-in; historical defaults/results are preserved.",
                   "test1534 has now been used for harness diagnosis; it is not untouched confirmation for the new harness."]}
    inputs = [OUT / n / "summary.json" for n in audits] + [OUT / "oracle_probe50/summary.json",
        OUT / "scoring_contract/summary.json", OUT / "boundaries.json", OUT / "tests.log"]
    summary["evidence_sha256"] = {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}
    write_json(OUT / "assessment.json", summary)
    lines = ["# 3B 尝试总结与 harness 归因审计（2026-11-17）", "",
        "本报告区分真实模型评测、Gold 辅助诊断与评分口径重标。未进行 SFT/RL，未修改权重或生成训练数据。", "",
        "## 结论", "",
        "当前瓶颈是跨库的 SQL 语义/值处理和信息使用；harness 存在可证的检索接口、观测与评分口径问题，但没有证据支持它把任务可达率压在当前 EX 附近。SFT 有开发集收益、留出收益较弱；RL 尚无可确认的任务增益。", "",
        "“工具选择最优仍失败”不能自动排除模型/训练问题：如果 SQL 仍由模型生成，模型仍可能误解问题、选错聚合/字面量。只有把正确 SQL 本身也作为 oracle 动作，仍被拒绝或执行/评分失败，才能定位对应题目的机械阻塞。", "",
        "## 已有训练尝试（实测，保留当时评分口径）", "",
        "| 实验 | 结果 | 可支持的结论 |", "|---|---|---|",
        "| 旧 compact SFT | dev200 49.5%；test1534 30.70% | 可工作的历史起点 |",
        "| M6-08i E1 专项 LoRA | dev200 39.5% vs 49.5%，p=0.0005 | 工具使用与列处理退化，候选失败 |",
        "| M6-08j/k 数据修订 | 未通过数据关卡，未训练 | 不证明 SFT 不可行，也不能当成功数据交付 |",
        "| M6-08m 自采＋full-FT，lr 2e-5 | epoch3 dev200 41.0% | 训练 loss 下降不保证任务改善 |",
        "| 同数据低 LR 5e-6 | 选 step788，dev200 58.0%；test1534 32.66% vs 30.70%，p=0.0675，DB CI 跨零 | 开发收益明显，跨库收益弱；继续训练后回落 |",
        "| R1 terminal GRPO | dev200 step10/20 58.5% vs R0 58.0%，p=1.0 | 当前中性，不能归为已提升 |",
        "| R3 DAPO DS | 训练完成20步；现有 step10 dev200 58.5%；step20完整评测文件未找到 | 增加 mixed 组未在现有评测上形成明确收益；不补造 step20 结果 |", "",
        "来源：`reports/m6/e1_alignment/assessment.md`、`reports/m6/e1_student_only/sft_lowlr_dev200_results.md`、`rl_r1_step20_paired.json`、`rl_r3_step10.json`、`rl_r3_training_summary.json`；旧任务状态见 `docs/TASKS_M6.md`。", "",
        "## 机械可达性与信息可达性（新增实测）", "",
        f"- dev200：真实 episode/parser/预算路径可提交 Gold {dev['gold_submitted_exactly']}/{dev['n']}，真实 SQL worker 自评分 {dev['gold_self_correct']}/{dev['n']}；最多两次 Gold 引导 lookup 的有限搜索找到完整物理表列证据 {dev['finite_lookup_search_schema_complete']}/{dev['n']}。",
        f"- test1534：Gold 可提交 {test['gold_submitted_exactly']}/{test['n']}；Gold 自评分 {test['gold_self_correct']}/{test['n']}。直接照抄 Gold 的已见阻塞为一条 SQL 超 action token 上限，另两条 Gold 自身执行超时。",
        "- 超长 SQL `bird_dev_1016`：复用已有 CTE 后 547→407 action tokens；同原 Gold 执行评分正确。因此 Gold 文本超长不等于没有预算内的等价解。两条超时题 `bird_dev_518/701` 延长至120秒仍不能执行 Gold，属于基准执行问题。证据 `boundaries.json`。",
        f"- E1.1＋compact 的有限检索搜索覆盖 {test['finite_lookup_search_schema_complete']}/{test['n']}；E1.2＋compact_v2 覆盖 {fixed['finite_lookup_search_schema_complete']}/{fixed['n']}。这是 Gold 选查询的离线工具验收，绝不是模型 EX 增益，也不是全局 oracle 上界。", "",
        "Gold 表列依赖用 SQLite authorizer 在 EXPLAIN 编译期提取，支持空格列名/别名/子查询；覆盖只认模型实际 Observation 文本，不认截断前 metadata。Gold 依赖全集并非 EX 正确的必要条件（等价 SQL 可用不同路径），也不足以保证语义与值推理正确。", "",
        "来源：`dev200_e1_1/summary.json`、`test1534_e1_1/summary.json`、`test1534_e1_2/summary.json`；逐题现场在相应目录 `cases.jsonl`。", "",
        "## 固定模型的信息增强对照（新增实测）", "",
        f"- step788，dev200 按 DB 比例抽样 seed17，{probe['completed_pairs']} 对完整完成；同一权重、同一 greedy/工具预算/评分，改变的是额外提供 Gold 选中的真实 schema 与描述，不提供 Gold SQL/结果/字面量。",
        f"- 当前 harness 正确 {probe['current_correct']}/{probe['completed_pairs']}；oracle schema 正确 {probe['oracle_correct']}/{probe['completed_pairs']}；gain {probe['gain']} / loss {probe['loss']}，McNemar p={probe['mcnemar_exact_p']:.4f}；两边都错 {probe['both_wrong']}，oracle 下可执行但错 {probe['oracle_valid_but_wrong']}。",
        f"- 当前臂与历史同题 correct 一致 {probe['historical_current_correct_agreement']}/{probe['completed_pairs']}，final SQL 完全一致 {probe['historical_current_final_sql_exact_agreement']}/{probe['completed_pairs']}；infra {probe['infrastructure_errors']}。",
        "小样本、Gold 特权信息、额外上下文格式均限制解释。该结果不能证明所有 harness 改进都无效，也不是‘理想模型上界’；它直接说明给更多正确 schema 并不保证现有模型产生正确 SQL。", "",
        "具体错误现场（见 oracle_probe50/episodes.jsonl）：最负债客户使用 MAX(c_acctbal) 而非 MIN；德国 HOUSEHOLD 客户把大写真实取值写成 Germany/household；多个菜品使用标量子查询等号，只保留其中一个；已给出 app_events.is_active 仍引用 app_labels.is_active。值大小写例还涉及真实值发现，不能全部归成纯 SQL 推理。", "",
        "## 评分口径确实存在偏差（新增实测）", "",
        "项目历史 `bird-set-v3` 实际要求相同行数＋规范化行集合；官方 BIRD Mini-Dev `calculate_ex` 只比较原始行集合。额外行数条件是项目自己的更严格指标，不应不加区分地叫官方 BIRD EX。", "",
        "官方源码：https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py （calculate_ex）。", "",
        "| 已保存预测 | 历史严格正确 | set-only 额外判对 | 重标后的集合正确数（非新模型运行） |",
        "|---|---:|---:|---:|"]
    for name, s in relabeling.items():
        lines.append(f"| {name} | {s['project_strict_correct']}/{s['n']} | {s['set_only_additional_correct']} | {s['derived_set_only_correct']}/{s['n']} ({s['derived_set_only_ex']:.2%}) |")
    lines.extend(["", "重标仅复核历史 valid-but-wrong；没有复跑新模型/完整官方评测。重复行被官方 set-only 忽略，不代表自然语言语义一定正确。两套指标应分开报告；切换评分不是模型能力提升。历史 reward 没有被静默改写。逐题评分在 `scoring_contract/`。",
        f"重标后同批 base→SFT 配对 gain {gain}/loss {loss}，McNemar p={relabeled_pair['mcnemar_exact_p']:.4f}，DB bootstrap 95% CI {relabeled_pair['db_bootstrap_ci95']} 仍跨零。不能直接沿用历史严格口径的 p 值；也不能把口径修正写成新训练收益。", "",
        "## 错误规模与已修复项", "",
        f"step788 留出集：历史严格正确 {history['correct']}；valid-but-wrong {history['valid_but_wrong']}；其余不满足可执行评分 {history['invalid_sql']}（含 infra {history['infrastructure_errors']}）。Gold 物理表列已在实际 schema Observation 出现但仍错 {history['wrong_with_schema_visible']}。这些是描述性分解，不是互斥因果归因。",
        f"旧列需求正则漏掉至少一个实际 Gold 依赖列的题目有 {history['legacy_requirements_omit_sqlite_columns']}/{history['n']}；不能再用旧列 recall 证明信息充分。E1.1 有 {test['cases_with_metadata_overstatement']} 题在本次候选查询中出现 metadata 高估，{test['cases_with_wrapper_truncation']} 题出现 wrapper 截断。", "",
        "代码修复：", "",
        "- `eval/sql_dependencies.py`、`eval/gold_extraction.py`：用 SQLite 编译读依赖替代正则需求提取；未解析明确标注。",
        "- `eval/evaluate_agent.py`：新增 `usage_visible`，原 usage 明确标为截断前 metadata；修复 join 端点错误交换产生的假命中。",
        "- `retrieval/e1_2.py`、`tools/schema_lookup.py`：显式支持 table/column 精确检查，未知列返回错误；表级请求不再先硬截前十列。",
        "- `agent/sql_episode.py`：新增 compact_v2，按完整 JSON 实际预算裁剪、保持可解析包装，避免400字符断崖。",
        "- SQL worker/reward/eval：新增显式 `bird_set`，保留 `project_strict` 历史默认及版本标签。", "",
        "新模式可通过 `eval/evaluate_agent.py --retrieval-version E1.2 --observation-mode compact_v2 --score-mode bird_set ...` 显式选择。后续新模型比较必须两臂同配置，并把评分契约写入数据/训练 fingerprint。RL 不应直接拿新口径曲线接旧 reward 曲线。", "",
        "## 当前应停止混淆的结论", "",
        "工程：已定位并修复可证缺陷，补齐 Gold 可达性、真实可见信息、评分与固定模型对照。实际模型上的 E1.2 EX 收益尚未验证。",
        "学习信号：原 SFT 自采成功筛选与 terminal reward 尚不能稳定覆盖困难语义；本次评分差异还说明部分历史失败在官方规则下是成功。不能直接推断过程奖励或更多训练步数会解决。",
        "任务能力：SFT 的跨库收益仍弱，已有 RL 无明确新增收益；没有证据把主要失败全部归给 harness，也不能宣布训练无关。优先分别测真实信息发现和语义/值处理，避免同时更换模型、harness、数据、reward 后失去归因。", "",
        "本次已用 test1534 做 Gold 辅助 harness 诊断；之后新 harness 的该集合成绩必须披露此用途，不能再称未见确认。历史已完成 SFT 比较保留其当时用途。", "",
        "## 验证与复现", "", "```bash",
        "python3 scripts/audit_text2sql_harness.py --data dev200=artifacts/datasets/dev200.jsonl --out reports/m6/harness_audit_20261117/dev200_e1_1",
        "python3 scripts/audit_text2sql_harness.py --data test1534=data/processed/bird_agent/test.jsonl --retrieval-version E1.2 --observation-mode compact_v2 --skip-sql --out reports/m6/harness_audit_20261117/test1534_e1_2",
        "# NPU 配对诊断：需正确加载 CANN 环境，使用新的输出目录，默认30分钟墙钟上限",
        "python3 scripts/probe_harness_oracle.py --limit 50 --max-seconds 1800 --out reports/m6/harness_audit_20261117/oracle_probe50_rerun",
        "```", "", "测试实测：", "", "```text", (OUT / "tests.log").read_text().strip(), "```", "",
        "完整证据索引、配置及哈希见 `assessment.json` 与各 summary.json。"])
    (OUT / "assessment.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"oracle_probe50": probe, "report": str(OUT / "assessment.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
