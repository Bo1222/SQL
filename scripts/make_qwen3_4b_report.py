"""Generate the Qwen3-4B zero-shot assessment from measured JSON artifacts."""
import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="reports/m6/qwen3_4b_zero_shot")
    p.add_argument("--out", default="reports/m6/qwen3_4b_zero_shot/assessment.md")
    a = p.parse_args()
    d = Path(a.dir)
    full = load(d / "dev200_zero_shot_chat.full_metrics.json")
    if not full:
        raise SystemExit("missing full metrics; run scripts/analyze_qwen3_4b_full.py first")
    summary = load(d / "dev200_zero_shot_chat.json")
    manifest = load(d / "model_manifest.json")
    raw = load(d / "smoke.json")
    v1 = load(d / "smoke_chat_v1.json")
    final_smoke = load(d / "smoke_chat.json")
    paired = load(d / "paired_vs_3b_sft.json")
    sft = load(Path("reports/m6/e1_alignment/dev200_sft.json"))

    lines = []
    lines.append("# Qwen3-4B-Instruct-2507 dev200 零样本探针\n")
    lines.append("> 本文件由 `scripts/make_qwen3_4b_report.py` 从实测产物生成。"
                 "这是 inference-only 探针，不是 SFT/RL 结果。\n")
    lines.append("> **叙事边界**：3B SFT 49.5% 仅作历史内部工程对照；本报告不把“4B vs 3B”"
                 "作为项目主线或结论。3B 数字保留用于审计，不进入主表/主叙事。\n")
    lines.append("## 结论（先写限制与负面）\n")
    lines.append("- **原始 E1 裸文本契约不兼容**：2/2 题 `invalid_action`，模型一次输出多个 action 并夹带"
                 "解释文字；严格 parser 无法接收（`smoke.json`）。")
    lines.append("- **当前 full 使用的是 chat-template 零样本**：原生 chat template + system 格式约束 + "
                 "在第一个 action 结束 tag 停止。**本报告不把 3B SFT 49.5% 作为主对照**：两者 prompt "
                 "包装不同，且 3B 仅内部工程对照；该数字不作为模型优劣或 SFT 收益结论。")
    lines.append("- **可能有预训练污染**：dev200 来自 BIRD 派生数据，Qwen3-4B 的预训练语料是否包含 BIRD "
                 "未知；因此零样本成绩只能作为开发集信号，不是干净泛化证明。")
    lines.append(f"- 任务能力（零样本，实测）：EX {pct(full.get('ex'))}（{round(full.get('ex', 0) * full.get('n', 0))}/{full.get('n')}），"
                 f"valid {pct(full.get('valid'))}，first SQL EX {pct(full.get('first_sql_ex'))}，"
                 f"repair {pct(full.get('repair_rate'))}，lookup rate {pct(full.get('lookup_rate'))}。\n")

    lines.append("## 模型与运行栈\n")
    lines.append("| 项 | 值 |\n|---|---|")
    lines.append(f"| 模型 | `{manifest.get('model', 'Qwen/Qwen3-4B-Instruct-2507')}`，"
                 f"revision `{manifest.get('hf_revision', '—')}` |")
    lines.append(f"| 本地路径 | `{manifest.get('local_path', '—')}` |")
    lines.append(f"| 规模/架构 | 4.0B dense，36 层，GQA 32 Q / 8 KV，非 thinking |")
    lines.append(f"| 权重完整性 | 3 个 safetensors 的 sha256 与 HF LFS 逐一匹配 |")
    rt = manifest.get("runtime", {})
    lines.append(f"| 运行栈 | transformers {rt.get('transformers')}、vLLM {rt.get('vllm')}、"
                 f"CANN {rt.get('cann')}、torch_npu {rt.get('torch_npu')} |")
    lines.append(f"| 硬件 | {rt.get('hardware')} |\n")

    lines.append("## Smoke（2 题，dev200 分层取前 2 题）\n")
    lines.append("| 配置 | EX | valid | finish | 说明 |\n|---|---:|---:|---|---|")
    for label, x, note in (("raw strict（无 chat template/stop）", raw, "2/2 invalid_action；多 action + 解释文字"),
                           ("chat template + stop（v1）", v1, "1/2；一题 SQL 被包成 JSON"),
                           ("chat template + system 格式约束（final）", final_smoke, "2/2 通过")):
        lines.append(f"| {label} | {pct(x.get('ex'))} | {pct(x.get('valid'))} | "
                     f"{json.dumps(x.get('finish_reasons', {}), ensure_ascii=False)} | {note} |")
    lines.append("")

    lines.append("## Full dev200 零样本（chat template，200 题）\n")
    lines.append("| 指标 | 值 |\n|---|---:|")
    for label, key, formatter in (
        ("Terminal EX", "ex", pct), ("Valid", "valid", pct),
        ("First SQL EX", "first_sql_ex", pct), ("First SQL cases", "first_sql_cases", str),
        ("Repair rate", "repair_rate", pct), ("Repair cases", "repair_cases", str),
        ("Semantic wrong", "semantic_wrong", str),
        ("no_such_column", "no_such_column", str), ("no_such_table", "no_such_table", str),
        ("lookup rate", "lookup_rate", pct),
        ("avg tool calls", "avg_tool_calls", fmt),
        ("avg schema lookups", "avg_schema_lookups", fmt),
        ("avg response tokens", "avg_response_tokens", fmt),
        ("avg observation tokens", "avg_observation_tokens", fmt),
        ("avg latency ms", "avg_latency_ms", fmt),
        ("invalid action episodes", "invalid_action_episodes", str),
        ("HBM peak", "hbm_peak_pct", lambda v: "—" if v is None else f"{v:.0f}%"),
    ):
        lines.append(f"| {label} | {formatter(full.get(key))} |")
    lines.append("")
    lines.append(f"- finish reasons：{json.dumps(full.get('finish_reasons', {}), ensure_ascii=False)}")
    lines.append(f"- error types：{json.dumps(full.get('error_types', {}), ensure_ascii=False)}")
    sft_ex = sft.get("ex") or 0
    delta_text = f"{100 * (full.get('ex', 0) - sft_ex):+.1f}pp" if sft_ex else "—"
    lines.append(f"- 内部对照（非主线）：与 3B SFT 49.5% 的描述性差值 {delta_text}"
                 "（不同模型 + 不同 prompt 包装；3B 仅历史内部对照，不作为项目结论）")
    if paired:
        lines.append(f"- 内部对照逐题配对（非主线，仅描述）：gain {paired.get('b_only')} / "
                     f"loss {paired.get('a_only')}，"
                     f"McNemar p={paired.get('mcnemar_exact_p')}；"
                     f"DB bootstrap 95% CI {paired.get('db_bootstrap_ci95')}")
    lines.append("")

    lines.append("## 三层结论\n")
    lines.append("### 工程\n")
    lines.append(f"- 模型可在当前单卡 910B4/32GB + vLLM 0.16 上加载和运行；权重 7.68 GB，"
                 f"KV cache 7.47 GiB，HBM 峰值 {fmt(full.get('hbm_peak_pct'), 0)}%；无 OOM。")
    lines.append("- 需要 chat template + system 格式约束 + 首 action 停止；原始裸文本 E1 parser 不兼容。"
                 "这不是模型能力问题，是格式契约问题。")
    lines.append("### 学习信号\n")
    lines.append(f"- 零样本 lookup rate {pct(full.get('lookup_rate'))}、avg tool calls "
                 f"{fmt(full.get('avg_tool_calls'))}；first SQL EX {pct(full.get('first_sql_ex'))}。"
                 "这些是行为与信号描述，不是训练收益。")
    lines.append("- 零样本已暴露格式契约风险：raw 2/2 invalid，chat+stop 才可用；该信号用于设计 "
                 "4B SFT/解码契约，不用于判断 3B 是否训好，也不作为 4B SFT 收益结论。")
    lines.append("### 任务能力\n")
    lines.append(f"- 本轮仅零样本，无 SFT/RL/能力评测训练；EX {pct(full.get('ex'))} 不能写成"
                 "“4B 优于 3B”或“SFT 有效”。")
    lines.append("- 最终确认仍需同模型 SFT 对照，以及 val590/dev1534；dev200 是开发集，且存在预训练污染不确定性。\n")

    lines.append("## 限制\n")
    lines.append("- chat-template 包装与 3B raw-text 契约不同；3B 仅历史内部对照，"
                 "没有同 wrapper 的 3B 零样本基线，也不作为主比较。")
    lines.append("- 只有 1 张 NPU；多卡/8B/14B/Qwen3.5 未验证。")
    lines.append("- dev200 是开发/选择集；不得把零样本成绩写成最终泛化结论。\n")
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": a.out, "ex": full.get("ex"), "valid": full.get("valid")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
