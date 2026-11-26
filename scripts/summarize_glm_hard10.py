"""Summarize the authorized, diagnostic-only GLM hard-question pilot."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/m6/glm_hard10_20261123'
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def main():
    trace_summary=json.loads((OUT/'trace_masked'/'trace_summary.json').read_text())
    merge=json.loads((OUT/'merge_summary.json').read_text())
    rows=read(OUT/'compatible/records_final.jsonl')
    assert len(rows)==10 and len({r['id'] for r in rows})==10
    assert sum(r['selected_from']!='main' for r in rows)==len(merge['selected_from_retry'])
    audits={r['id']:r for r in read(OUT/'audit.jsonl')}
    clean=[r for r in rows if r['audit_status'] in ('supported','supported_with_scope_note')]
    assert len(clean)==5
    calls=[c for r in rows for c in r['api_calls']]
    assert all(c['request']['model']=='WAIC-GLM-5.3-Flash' for c in calls)
    assert all(r['training_eligible'] is False for r in rows)
    assert all(r['rounds']<=5 and r['tool_calls']<=4 and r['schema_lookups']<=2 and r['sql_executes']<=2 for r in rows)
    assert all(c['request']['max_tokens']==4096 for c in calls)
    visible_overruns=sum(c['visible_student_tokens']>c['budget'] for c in calls)
    summary={'n':10,'correct_gold_set':sum(r['correct'] for r in rows),'submitted':sum(r['finish_reason']=='answer_submitted' for r in rows),
        'clean_subset_n':5,'clean_subset_correct':sum(r['correct'] for r in clean),
        'finish_reasons':dict(Counter(r['finish_reason'] for r in rows)),
        'unavailable':sum(not r['reward_available'] for r in rows),'api_calls':len(calls),
        'reported_completion_tokens':sum((c['usage'] or {}).get('completion_tokens',0) for c in calls),
        'reported_reasoning_tokens':sum(((c['usage'] or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) for c in calls),
        'returned_models':sorted({str(c['model_returned']) for c in calls}),
        'visible_budget_overrun_calls':visible_overruns,'no_sft_export':True,
        'retried_from_main_protocol_failure':merge['retried'],'selected_from_retry':merge['selected_from_retry'],
        'unavailable_protocol_after_merge':merge['unavailable_protocol'],
        'retry_rule':'Retried only questions whose main attempt ended in a provider protocol failure (transport error or empty visible action); the retry outcome replaced the main one under a rule that never looks at correctness.',
        'limits':['Selected training questions, not an independent benchmark.','One teacher trajectory per question; cannot establish a capability ceiling.',
            'Same information, tools and visible budgets; provider reasoning ceiling and action-stop decoder differ from the student.',
            'Three clear label conflicts, one date-format conflict and one output-layout ambiguity were identified before teacher outputs.']}
    (OUT/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# GLM 教师可解性探针','', '2026-11-23。指定模型 WAIC-GLM-5.3-Flash；端点由用户提供，凭据从 .env 的 GLM_API_KEY 读取，未落盘。','',
      '## 审阅与结果','',f"10 题原始 Gold set-only 正确 {summary['correct_gold_set']}/10；提交答案 {summary['submitted']}/10。事前审阅中暂未发现明确冲突的 5 题，正确 {summary['clean_subset_correct']}/5。其余题的原 Gold 分数保留，不能把标签/布局差异直接解释成教师能力错误。",'',
      '| DB / id | 事前审阅 | 结束原因 | Gold EX | 采用 |','|---|---|---|---:|---|']
    for r in rows:lines.append(f"| {r['db_id']} / {r['id']} | {r['audit_status']} | {r['finish_reason']} | {int(r['correct'])} | {r['selected_from']} |")
    reasons={
      'bird_train_42106bdd27ea14b7197d':'提交姓名+区域，Gold 只投影区域；事前审阅已判为 Gold 漏投影。',
      'bird_train_7d760291cd5d733d8145':'按 Evidence 用 Restaurant 过滤，Gold 用 Liquor；事前审阅已判为 Gold/Evidence 冲突。',
      'bird_train_0d0e403a1a7d3c0b127e':'按 Evidence 用 EmailPromotion=0（实际 162），Gold 用 1（61）；事前审阅已判冲突。该题第 2、4 轮响应为纯说明文字，被严格单动作解析器两次计违规后终止，其间第 3 轮已执行并得到 162，未及提交。',
      'bird_train_09fe02c588acd2b550f1':'照 Evidence 字面量 1/1/2018 过滤（7972 行），Gold 以 TEXT 比较 1/1/18（7991 行），差 19 行；事前审阅已判日期格式冲突。',
      'bird_train_0e6f6535a44f3a88fa09':'分组返回 High/Low/Medium 三行（3/84/10），Gold 为单行三列透视；数字一致但布局不同，事前审阅已判布局歧义。',
      'bird_train_9381242a653b741b840a':'第 3 轮为说明文字+重复查询、第 5 轮为纯说明文字，均被严格单动作解析器计违规；第 4 轮执行已返回 100.0、与 Evidence 公式一致，但 5 轮用尽未提交。',
    }
    lines+=['','## 逐题归因（非正确项）','']
    for r in rows:
        if not r['correct']:
            lines.append(f"- {r['db_id']} / {r['id']}：{reasons.get(r['id'],'见 audit.jsonl')}")
    lines+=['',
      '两处 invalid_action 的成因都是响应只含说明文字（或说明+动作），不是 SQL 报错：beer_factory 与 works_cycles 都在被终止前已经执行出与 Evidence 口径一致的数值（100.0 与 162），但都没有走到 answer。严格单动作协议、5 轮与 2 次违规上限把这类冗长响应直接判负；这属于协议遵守差异，不是 SQL 能力证据，也不是服务端协议失败。']
    lines+=['','## 协议失败与重跑','',
      f"主运行 10 题中有 {len(merge['retried'])} 题以服务端协议失败收尾：regional_sales 第 2 次响应 finish=stop 但可见内容为空（completion 1131 token 全为隐藏推理），public_review_platform 第 4 次请求传输异常。两者都不是候选侧能力证据，因此在完全相同条件下重跑这两题（temperature、输入、工具、可见预算、上下文与 max_tokens 均不变），重跑产物单独归档为 records_retry1.jsonl / api_responses_retry1.jsonl，合并规则只依据协议失败、不看对错，写入 records_final.jsonl。",
      '重跑后两题都提交了答案，且失败原因与事前审阅一致：日期字面值冲突与输出布局歧义。合并后 10 题全部有候选侧结论，不再有 unavailable。', '',
      '## 接口适配与 thinking','',
      'API 已连通，返回的模型名与请求一致。初次 max_tokens=512 的调用将 512 token 全用于隐藏推理，content 为空。请求 thinking disabled 后 usage 仍报告 reasoning_tokens，因此不能认为该关闭参数生效。原始响应保存在 provider_default/ 及根目录 records.jsonl；未并入正式适配组。',
      '适配组保留原 A-e1.2-typed-v1 输入、E1.2/compact_v2、5轮/4工具/各2次、可见动作512、观测1024、总响应4096和上下文8192。服务端 max_tokens=4096 允许隐藏推理；API stop 在动作闭合标签处停止，只恢复服务端省略的终止标签，不抽取多动作、不删除额外说明文字、不生成伪观察。',
      'hockey 已完成真实 schema_lookup → sql_execute → answer，提交 G=9 对应查询，与 Gold 一致。停止 token 与隐藏推理预算是显式适配差异，因此不是等算力比较。', '',
      f"适配组共 {summary['api_calls']} 次 API 响应，报告 completion token {summary['reported_completion_tokens']}，其中 reasoning token {summary['reported_reasoning_tokens']}；这些数字不包含中断请求的未知服务端消耗。可见动作超预算调用 {visible_overruns} 次；超限/格式错误保留为失败，不静默裁剪。",'',
      '## 教师轨迹（token 掩码，不做 prompt/生成端约束）','',
      f"同一批 10 题、同一 prompt 与工具循环，改为「服务端原样返回 → 只在第一个完整动作块上保留 token」的重放：10/10 提交，33 次响应全部解析成动作（0 次无动作、0 次解析失败、0 次超预算），Gold set-only 正确 {trace_summary['correct']}/10，无冲突 5 题 {trace_summary['clean_subset_correct']}/5。",
      f"掩码保留动作 token {trace_summary['masked_action_tokens']}、丢弃说明文字 token {trace_summary['prose_tokens_dropped']}，导出 (context, action) 对 {trace_summary['sft_pairs']} 条（动作最长 {trace_summary['max_target_tokens']} token、上下文最长 {trace_summary['max_context_tokens']} token）。",
      '严格协议下教师因散文被判违规的两题（beer_factory、works_cycles）在掩码重放里都提交了答案；但两版都只是单样本（temperature 0.7，一题一条轨迹），5 道无冲突题里有 3 题在两个解码器之间互有翻转，差异不能归因给掩码。细节、逐题对照与动作层面对照见 trace_masked/TRACE.md。','',
      '## 使用边界','',
      '此探针不使用 held-out 20、dev200 或 test；Gold、审阅结论、原任务 full-schema prompt、Gold 选表列都不进入教师输入。程序不导出 SFT/RL 训练数据，未训练新模型。',
      '一次成功只证明该题出现了可用轨迹。协议失败、标签冲突和真正的 SQL 错误需要分开归因；不由本次少量结果宣布模型上限或 benchmark 增益。', '',
      '证据：AUDIT.md、audit.jsonl、audit_summary.json、supplementary_checks.json、compatible/manifest.json、compatible/records.jsonl（主运行）、compatible/api_responses.jsonl、compatible/records_retry1.jsonl 与 api_responses_retry1.jsonl（协议失败重跑）、compatible/records_final.jsonl（合并定稿）、merge_summary.json、final_summary.json。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix not in ('.log',)]
    (OUT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
