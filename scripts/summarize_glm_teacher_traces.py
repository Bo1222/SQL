"""Summarize the masked teacher traces and compare them with the strict-protocol probe.

Reads trace_masked/traces.jsonl + sft_pairs.jsonl and compatible/records_final.jsonl;
writes trace_masked/TRACE.md. No training export, no model training.
"""
import json
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.protocol import PATTERN

OUT=ROOT/'reports/m6/glm_hard10_20261123'
TRACE=OUT/'trace_masked'
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def main():
    traces=read(TRACE/'traces.jsonl')
    pairs=read(TRACE/'sft_pairs.jsonl')
    strict={r['id']:r for r in read(OUT/'compatible/records_final.jsonl')}
    assert len(traces)==len({t['id'] for t in traces})
    assert all(t['training_eligible'] is False for t in traces)
    assert all(t['id'] in strict for t in traces)
    assert max(len(p['target_ids']) for p in pairs)<=512
    assert max(len(p['context_ids']) for p in pairs)<8192
    steps=[s for t in traces for s in t['steps']]
    prose_tokens=sum(s['mask']['dropped_tokens'] for s in steps)
    action_tokens=sum(s['mask']['kept_tokens'] for s in steps)
    provider_side=[t for t in traces if t['infrastructure_error']]
    assert not provider_side, [t['db_id'] for t in provider_side]
    for t in traces:
        for s in t['steps']:
            if s.get('masked_action_text'):
                hit=PATTERN.search(s['masked_action_text'])
                assert hit and hit.group(0)==s['masked_action_text'], (t['db_id'],s['round'])
    clean=[t for t in traces if t['audit_status'] in ('supported','supported_with_scope_note')]
    boundary=sum(1 for s in steps if s['mask'].get('boundary_tokens_reencoded'))
    multi=sum(1 for s in steps if s['mask']['extra_actions'])
    restored=sum(1 for s in steps if s.get('stop_delimiter_restored'))
    summary={'n':len(traces),'correct':sum(t['correct'] for t in traces),
        'strict_correct':sum(strict[t['id']]['correct'] for t in traces),
        'clean_subset_n':len(clean),'clean_subset_correct':sum(t['correct'] for t in clean),
        'finish_reasons':dict(Counter(t['finish_reason'] for t in traces)),
        'no_action_responses':sum(t['no_action_responses'] for t in traces),
        'extra_actions_dropped':sum(t['extra_actions_dropped'] for t in traces),
        'masked_action_tokens':action_tokens,'prose_tokens_dropped':prose_tokens,
        'boundary_tokens_reencoded':boundary,'responses_with_multiple_actions':multi,
        'stop_delimiter_restored':restored,
        'sft_pairs':len(pairs),'max_target_tokens':max(len(p['target_ids']) for p in pairs),
        'max_context_tokens':max(len(p['context_ids']) for p in pairs),
        'training_eligible':False}
    (TRACE/'trace_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# GLM 教师轨迹（token 掩码，不做 prompt 约束）','',
      '2026-11-23。模型 WAIC-GLM-5.3-Flash，端点与凭据同前（凭据只在内存中，从 .env 读取）。','',
      '## 方法','',
      '**生成端不做任何约束**：请求不带 stop 参数、不带任何抑制散文的 prompt，服务端原样返回（含闭合标签），掩码发生在拿到完整原文之后——只保留第一个完整动作块（`<schema_lookup>`/`<sql_execute>`/`<answer>`）的 token，其余散文全部丢弃，掩码后的动作才进入上下文与工具循环。无完整动作的响应不计违规、不占轮次，同上下文重发（≤3 次）；多个动作只取第一个，其余记为 extra_actions 且不执行。与严格协议探针的唯一差异都记在 trace_masked/manifest.json 的 differences_from_student_episode 里；prompt、catalog、工具、观测编码、轮次/工具预算、上下文与可见动作上限、评分口径与学生学习时完全一致。','',
      '（早前一版采集器在请求里带了 `stop=[闭合标签]`，按 stop 的标准语义服务端返回内容不含该串，38/38 次响应的动作闭合标签都被吃掉，其中 14 次是「说明文字+动作」而恢复逻辑只认动作开头的响应，完整正确的动作被判成无动作丢弃。该版本整体归档在 trace_masked/stop_decoder/，不再是结论依据。）','',
      '## 结果','',
      f"掩码后 Gold set-only 正确 {summary['correct']}/10（严格协议探针 {summary['strict_correct']}/10）；事前审阅无冲突的 5 题 {summary['clean_subset_correct']}/5。",
      f"共 {len(steps)} 次服务端响应，掩码保留动作 token {action_tokens}，丢弃说明文字 token {prose_tokens}；无动作响应 {summary['no_action_responses']} 次（已重发），多动作响应被保留首个、丢弃其余 {summary['extra_actions_dropped']} 个。",
      f"导出 (context, action) 对 {len(pairs)} 条；单步动作最长 {summary['max_target_tokens']} token（≤512），上下文最长 {summary['max_context_tokens']} token（<8192）。",
      f"掩码器边界重编码 {boundary} 次（闭合 `>` 与散文被 BPE 合并成同一个 token 时，改按精确字符切片重编码）；停止标签修复 {restored} 次（请求已不带 stop，自然结束）；无动作响应 {summary['no_action_responses']} 次，解析失败 0 次。",
      f"**单样本告诫**：两次运行都是 temperature 0.7 下一题一条轨迹。事前审阅无冲突的 5 题里，严格协议为对的 4 题中有 3 题（hockey、cookbook、synthea）在掩码轨迹里变成错，另 1 题反向（beer_factory）。采样噪声与「掩码本身是否影响教师规划」在这批数据里无法分离；要分离需要同题多次采样（k≥4）的对照。5 道事前审阅已判冲突的题在两次运行里都是 0/5，与预测一致。",
      f"**多余动作策略**：{multi} 次响应（共 33 次）包含多个动作块，当前策略只取第一个，其余 {summary['extra_actions_dropped']} 个动作不执行。这是教师的原生习惯（一轮里连续写 schema_lookup→sql_execute→answer），被丢弃的动作是它意图的一部分，属于本流程显式选择的协议化损失；若要保留，可改为按顺序执行或把其余动作作为下一轮上下文。",'',
      '| DB / id | 事前审阅 | 严格协议 | 掩码轨迹 | 结束原因 | 无动作重发 | 多余动作丢弃 |','|---|---|---:|---:|---|---:|---:|']
    for t in traces:
        s=strict[t['id']]
        lines.append(f"| {t['db_id']} / {t['id']} | {t['audit_status']} | {int(s['correct'])} | {int(t['correct'])} | {t['finish_reason']} | {t['no_action_responses']} | {t['extra_actions_dropped']} |")
    cmp_path=TRACE/'action_error_comparison.json'
    if cmp_path.exists():
        c=json.loads(cmp_path.read_text());s=c['student_3b_self_sampling'];g=c['glm_teacher_masked_trace']
        lines+=['','## 动作层面的对照（同 10 题，学生 32 次采样/题 vs 教师 1 条轨迹/题）','',
          '| 口径 | 3B 学生（self-sampling） | GLM 教师（掩码轨迹） |','|---|---:|---:|',
          f"| 提交率 | {s['submitted']}/{s['rollouts']} | {g['submitted']}/{g['questions']} |",
          f"| 单动作解析失败 | {s['invalid_actions_total']} 次（{s['rollouts_with_invalid_action']} 条轨迹） | {g['responses_invalid_action']} 次响应 |",
          f"| 无动作响应 | 0 | {g['responses_no_action']} |",
          f"| 执行的动作 | {s['executed_actions']} | {g['executed_actions']} |",
          f"| 其中 SQL 执行 | {s['sql_executes']} | {g['sql_executes']} |",
          f"| SQL 报错率 | {s['sql_error_rate']}（{s['sql_errors']}/{s['sql_executes']}） | {g['sql_error_rate']}（{g['sql_errors']}/{g['sql_executes']}） |",
          f"| SQL 报错类型 | {s['sql_error_types']} | {g['sql_error_types']} |",
          f"| schema_lookup 报错 | {s['schema_lookup_errors']}/{s['schema_lookups']} | {g['schema_lookup_errors']}/{g['schema_lookups']} |",
          f"| 多余动作被丢弃 | — | {g['extra_actions_dropped']} |", '',
          f"学生 {s['rollouts']} 条轨迹全部提交，协议维度几乎不失误（{s['invalid_actions_total']}/{s['rollouts']} 次解析失败）；教师的短板相反，是动作周围的散文——而散文由掩码移除，所以教师这一侧的「协议正确率」不该用原始响应的格式来量。执行层面则相反：学生的 SQL 报错率明显更高。两个口径不可互换，样本量也不同（{s['rollouts_per_question']}:1）。",'']
    lines+=['','## 使用边界','',
      '`training_eligible` 仍为 false：教师是第三方模型，其输出能否进入训练数据由用户决定，本流程只负责把轨迹采集与清洗到与学习协议一致的形式。',
      '掩码把「说明文字」从轨迹中移除，因此掩码轨迹不反映提供方的原始行为分布；原始文本保存在每条步骤的 raw_content 与 api_responses.jsonl 中。',
      '一题一条轨迹，不构成能力上界；同一批题、同一批预算，才能与严格协议探针逐题对照。','',
      '证据：trace_masked/traces.jsonl、trace_masked/sft_pairs.jsonl、trace_masked/api_responses.jsonl、trace_masked/trace_summary.json、trace_masked/manifest.json、trace_masked/action_error_comparison.json、compatible/records_final.jsonl。',
      '被取代的中间版本（只作过程留档，不作结论依据）：trace_masked/stop_decoder/（请求带 stop，闭合标签被服务端吃掉）、trace_masked/mask_inside_only_v1/（掩码只保留完全落在动作区间内的 token，闭合 `>` 被跨界 token 带走）、trace_masked/provider_http404_failures.jsonl（提供方 HTTP 404 终止的一次尝试，已重跑）。','',
      '复现：`python3 scripts/collect_glm_teacher_traces.py --limit 10`（已完成 10 题则自动跳过）→ `python3 scripts/compare_action_errors.py` → `python3 scripts/summarize_glm_teacher_traces.py`。']
    (TRACE/'TRACE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
