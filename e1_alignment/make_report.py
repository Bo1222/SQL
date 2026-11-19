"""Generate reports/m6/e1_alignment/assessment.md from measured artifacts."""
import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="artifacts/m6/e1_alignment/dataset_manifest.json")
    p.add_argument("--train", default="outputs/e1_alignment/e1_alignment_adapter_v2/train_summary.json")
    p.add_argument("--sft", default="reports/m6/e1_alignment/dev200_sft.json")
    p.add_argument("--v2", default="reports/m6/e1_alignment/dev200_v2.json")
    p.add_argument("--assessment", default="reports/m6/e1_alignment/assessment.json")
    p.add_argument("--out", default="reports/m6/e1_alignment/assessment.md")
    a = p.parse_args()

    manifest = load(a.manifest)
    train = load(a.train)
    sft = load(a.sft)
    v2 = load(a.v2)
    assess = load(a.assessment)
    paired = assess.get("paired", {})
    arms = assess.get("arms", {})
    sft_arm, v2_arm = arms.get("sft", {}), arms.get("v2", {})

    v2_ex = v2.get("ex")
    sft_ex = sft.get("ex")
    pvalue = paired.get("mcnemar_exact_p")
    gate = ("未通过（未同时满足 EX 提升与 p<0.1）" if (v2_ex is None or sft_ex is None
            or v2_ex <= sft_ex or pvalue is None or pvalue >= 0.1)
            else "通过同条件 EX + McNemar p<0.1 门槛（仍需检查 schema/恢复改善）")

    lines = []
    lines.append("# M6-08i E1 Tool-Environment Alignment SFT — Assessment\n")
    lines.append("> 本文件由 `e1_alignment/make_report.py` 从实测产物生成；所有数字均标注来源文件。\n")
    lines.append("## 结论（先写负面/门槛判断）\n")
    lines.append(f"- **放行门槛：{gate}。** V2 dev200 EX {pct(v2_ex)} vs 同条件 SFT control {pct(sft_ex)}；"
                 f"配对 gain {paired.get('gain', '—')} / loss {paired.get('loss', '—')}，"
                 f"McNemar p={fmt(pvalue)}（`{a.assessment}`）。\n")
    lines.append("## 1. 数据漏斗（E1-02–05，实测）\n")
    funnel = manifest.get("funnel", {})
    lines.append("| 来源 | 候选 | 正样本 | 重放通过 |\n|---|---:|---:|---:|")
    for source, values in funnel.items():
        lines.append(f"| {source} | {values.get('raw_candidates', '—')} | "
                     f"{values.get('positive_candidates', '—')} | {values.get('replay_passed', '—')} |")
    lines.append("")
    lines.append(f"- 入集 **{manifest.get('n', '—')}** 条；来源比例："
                 f"{json.dumps(manifest.get('source_ratio', {}), ensure_ascii=False)}")
    lines.append(f"- 去重：{json.dumps(manifest.get('dedupe', {}), ensure_ascii=False)}；"
                 f"scorer：{json.dumps(manifest.get('scorer', {}), ensure_ascii=False)}")
    lines.append(f"- 成功类型：{json.dumps(manifest.get('by_success_type', {}), ensure_ascii=False)}；"
                 f"B 失败类型：{json.dumps(manifest.get('by_failure_error_type', {}), ensure_ascii=False)}")
    lines.append(f"- 工具形态：{json.dumps(manifest.get('by_tool_shape', {}), ensure_ascii=False)}")
    lines.append(f"- Token：{json.dumps(manifest.get('token_stats', {}), ensure_ascii=False)}")
    lines.append(f"- 数据集 sha256 `{manifest.get('dataset_sha256', '—')}`；"
                 f"runtime fingerprint `{manifest.get('runtime_fingerprint_sha256', '—')}`\n")
    lines.append("## 2. SFT V2（E1-06，实测）\n")
    lines.append(f"- 数据：`{train.get('data', '—')}` sha256 `{train.get('data_sha256', '—')}`；"
                 f"kept {train.get('kept', '—')} / excluded {train.get('excluded', '—')}")
    lines.append(f"- 初始化：{train.get('initialization', '—')}；LoRA {json.dumps(train.get('lora', {}), ensure_ascii=False)}")
    lines.append(f"- 超参：epochs {train.get('epochs', '—')}、steps {train.get('steps', '—')}、"
                 f"grad_accum {train.get('grad_accum', '—')}、lr {train.get('lr', '—')}、"
                 f"max_context {train.get('max_context', '—')}；elapsed {train.get('elapsed_s', '—')} s")
    mask = train.get("mask_check", {})
    lines.append(f"- Mask 自检：labels 对齐 {mask.get('labels_align_with_segment_mask')}，"
                 f"masked loss {mask.get('masked_loss')} vs 全标签对照 {mask.get('unmasked_control_loss')}\n")
    lines.append("## 3. dev200 同条件对照（greedy, 6144, compact, E1）\n")
    lines.append("| 指标 | SFT control | V2 |\n|---|---:|---:|")
    metric_rows = [
        ("Terminal EX", "ex", pct), ("Valid", "valid_rate", pct),
        ("First SQL EX", "first_sql_ex", pct), ("First SQL cases", "first_sql_cases", str),
        ("Repair rate", "repair_rate", pct), ("Repair cases", "repair_cases", str),
        ("no_such_column episodes", "no_such_column_episodes", str),
        ("no_such_table episodes", "no_such_table_episodes", str),
        ("lookup rate", "lookup_rate", pct),
        ("observation grounded_all", ("schema_observation_use", "grounded_all_rate"), pct),
        ("observation grounded_any", ("schema_observation_use", "grounded_any_rate"), pct),
        ("avg tool calls", "avg_tool_calls", fmt),
        ("avg schema lookups", "avg_schema_lookups", fmt),
        ("avg response tokens", "avg_response_tokens", fmt),
        ("avg observation tokens", "avg_observation_tokens", fmt),
    ]
    for label, key, formatter in metric_rows:
        def value(arm):
            if isinstance(key, tuple):
                return (arm.get(key[0]) or {}).get(key[1])
            return arm.get(key)
        lines.append(f"| {label} | {formatter(value(sft_arm))} | {formatter(value(v2_arm))} |")
    lines.append("")
    lines.append(f"- 配对：n={paired.get('n', '—')}，gain {paired.get('gain', '—')}，"
                 f"loss {paired.get('loss', '—')}，ΔEX {fmt(paired.get('delta_ex'))}，"
                 f"McNemar p={fmt(paired.get('mcnemar_exact_p'))}，"
                 f"DB bootstrap 95% CI {paired.get('db_bootstrap_ci95')}")
    lines.append(f"- 转移矩阵：{json.dumps(paired.get('transitions', {}), ensure_ascii=False)}")
    lines.append(f"- gain 的 control 错误类型：{json.dumps(paired.get('gain_control_outcomes', {}), ensure_ascii=False)}；"
                 f"loss 的 control 结果：{json.dumps(paired.get('loss_control_outcomes', {}), ensure_ascii=False)}\n")
    lines.append("## 4. 三层结论\n")
    lines.append("### 工程\n")
    lines.append("- E1-01 契约、E1-02–05 数据管线、逐段 mask、SFT V2 和 dev200 对照均已执行；"
                 "数据有重放验证与来源标记。具体漏斗与成本见上表。")
    lines.append("### 学习信号\n")
    lines.append("- 数据侧：A 的 `success_type` 分布、B 的失败类型、C 的覆盖率见 `dataset_manifest.json`；"
                 "训练 loss/mask 见 `train_summary.json`。")
    lines.append(f"- 行为侧（dev200，实测）：lookup rate {pct(sft_arm.get('lookup_rate'))} → "
                 f"{pct(v2_arm.get('lookup_rate'))}；avg tool calls {fmt(sft_arm.get('avg_tool_calls'))} → "
                 f"{fmt(v2_arm.get('avg_tool_calls'))}；no_such_column episodes "
                 f"{sft_arm.get('no_such_column_episodes')} → {v2_arm.get('no_such_column_episodes')}；"
                 f"valid {pct(sft_arm.get('valid_rate'))} → {pct(v2_arm.get('valid_rate'))}。"
                 "这些是模型行为变化的证据，不是任务能力提升的证据。")
    lines.append("### 任务能力\n")
    if v2_ex is not None and sft_ex is not None and v2_ex < sft_ex:
        significant = pvalue is not None and pvalue < 0.05
        lines.append(f"- V2 vs control：EX {pct(sft_ex)} → {pct(v2_ex)}，配对 gain "
                     f"{paired.get('gain', '—')} / loss {paired.get('loss', '—')}，McNemar p={fmt(pvalue)}；"
                     f"这是{'统计显著的' if significant else '未达统计显著的'}退化，不能解释为增益或“无差异”。"
                     f"First SQL EX {pct(sft_arm.get('first_sql_ex'))}（{sft_arm.get('first_sql_cases')} 例）→ "
                     f"{pct(v2_arm.get('first_sql_ex'))}（{v2_arm.get('first_sql_cases')} 例）；Repair "
                     f"{pct(sft_arm.get('repair_rate'))}（{sft_arm.get('repair_cases')} 例）→ "
                     f"{pct(v2_arm.get('repair_rate'))}（{v2_arm.get('repair_cases')} 例），分母不同，不可直接比较。")
    else:
        lines.append(f"- V2 vs control：EX {pct(sft_ex)} → {pct(v2_ex)}；First SQL EX "
                     f"{pct(sft_arm.get('first_sql_ex'))} → {pct(v2_arm.get('first_sql_ex'))}；"
                     f"Repair {pct(sft_arm.get('repair_rate'))} → {pct(v2_arm.get('repair_rate'))}；"
                     f"no_such_column episodes {sft_arm.get('no_such_column_episodes')} → {v2_arm.get('no_such_column_episodes')}。"
                     "若未通过门槛，结论只能是“未观察到超过噪声的增益”，不能以 valid/lookup/loss 替代 EX。")
    lines.append("\n## 5. 限制\n")
    lines.append("- B 前缀动作文本为规范化重建；C 为 Gold-assisted，不能报告为自主成功。")
    lines.append("- 采集温度 1.0 与 greedy 评测不同；data mixture 不是调优后的最优配比。")
    lines.append("- dev200 为开发/选择集，非盲测；最终确认仍需 val590/dev1534。\n")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": a.out, "gate": gate, "sft_ex": sft_ex, "v2_ex": v2_ex,
                      "mcnemar_p": pvalue}, ensure_ascii=False))


if __name__ == "__main__":
    main()
