"""Summarize the 2c stratified expansion against its pre-registered gates.

GOAL.md §3 gates: (1) action parse rate >= 99%, (2) submission rate >= 95%,
(3) >= 60% of questions with >= 1 verified trace. Gate 3 uses the same
correct-and-execution-verified definition the 2d intake will formalize; here it
is a preview, so the report says so. Reading rules are GOAL.md §4: protocol
failure != label conflict != SQL capability, reported separately.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/m6/teacher_pool_20261123/expansion_50'
RETRYABLE = {'generation_error', 'no_action_limit', 'infrastructure_error'}
GATES = {'parse_rate': 0.99, 'submit_rate': 0.95, 'verified_question_rate': 0.60}
STRATUM_ORDER = ('1of4', '2of4', '3of4', 'all4')
STRATUM_LABEL = {1: '1of4', 2: '2of4', 3: '3of4', 4: 'all4'}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def is_failure(r):
    return bool(r.get('infrastructure_error')) or r.get('finish_reason') in RETRYABLE


def main():
    traces = read_jsonl(OUT / 'traces.jsonl')
    latest = {}
    for t in traces:
        latest[(t['id'], t.get('sample_idx', 0))] = t
    recs = list(latest.values())
    ids = sorted({r['id'] for r in recs})
    valid = [r for r in recs if not is_failure(r)]
    responses = sum(len(r.get('api_calls') or []) for r in valid)
    no_action = sum(r.get('no_action_responses', 0) or 0 for r in valid)
    parse_rate = 1 - no_action / responses if responses else None
    submitted = sum(r['finish_reason'] == 'answer_submitted' for r in valid)
    submit_rate = submitted / len(valid) if valid else None
    verified = sum(bool(r['correct'] and r['reward_available']) for r in valid)

    per_q = {}
    for qid in ids:
        eps = [r for (i, _), r in sorted(latest.items()) if i == qid]
        ok = [r for r in eps if not is_failure(r)]
        vec = {r.get('sample_idx', 0): int(r['correct'] and r['reward_available']) for r in ok}
        per_q[qid] = {
            'db_id': eps[0]['db_id'], 'audit_status': eps[0].get('audit_status'),
            'semantic_subclass': eps[0].get('stratum'),
            'stratum': STRATUM_LABEL.get(eps[0].get('stratum_of_4'), str(eps[0].get('stratum_of_4'))),
            'vector': [vec.get(k, None) for k in sorted(vec)],
            'k_valid': len(ok), 'k': len(eps), 'pass@k': int(any(vec.values())) if vec else 0,
            'submitted': sum(r['finish_reason'] == 'answer_submitted' for r in ok),
            'protocol_failure': sum(is_failure(r) for r in eps),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in ok)),
            'no_action': sum(r.get('no_action_responses', 0) or 0 for r in ok)}
    verified_questions = sum(v['pass@k'] for v in per_q.values())
    verified_rate = verified_questions / len(ids) if ids else None
    gates = {'parse_rate': {'value': parse_rate, 'threshold': GATES['parse_rate'],
                            'pass': parse_rate is not None and parse_rate >= GATES['parse_rate']},
             'submit_rate': {'value': submit_rate, 'threshold': GATES['submit_rate'],
                             'pass': submit_rate is not None and submit_rate >= GATES['submit_rate']},
             'verified_question_rate': {'value': verified_rate, 'threshold': GATES['verified_question_rate'],
                                        'pass': verified_rate is not None and verified_rate >= GATES['verified_question_rate']}}
    failures = [{'question_id': r['id'], 'db_id': r['db_id'], 'sample_idx': r.get('sample_idx'),
                 'finish_reason': r['finish_reason'], 'attempt': r.get('attempt'),
                 'reason': r.get('infrastructure_reason') or r.get('generation_error')}
                for r in recs if is_failure(r)]
    by_stratum, by_status = defaultdict(list), defaultdict(list)
    for v in per_q.values():
        by_stratum[v['stratum']].append(v)
        by_status[v['audit_status']].append(v)
    manifest = json.loads((OUT / 'manifest.json').read_text()) if (OUT / 'manifest.json').exists() else {}
    summary = {
        'scope': '2c stratified expansion; gate 3 is a preview of the 2d intake definition',
        'questions': len(ids), 'episodes': len(recs), 'episodes_valid': len(valid),
        'submitted': submitted, 'submit_rate': submit_rate,
        'responses': responses, 'no_action_responses': no_action, 'parse_rate': parse_rate,
        'correct_episodes_verified': verified,
        'episode_ex_rate': verified / len(valid) if valid else None,
        'questions_with_verified_trace': verified_questions, 'verified_question_rate': verified_rate,
        'protocol_failures': failures, 'unavailable': sum(not r['reward_available'] for r in valid),
        'gates': gates,
        'calls_spent': manifest.get('calls_spent'), 'call_budget': manifest.get('call_budget'),
        'budget_exhausted': manifest.get('budget_exhausted'),
        'by_stratum': {s: {'questions': len(v),
                           'episodes_valid': sum(x['k_valid'] for x in v),
                           'verified_episodes': sum(sum(1 for y in x['vector'] if y) for x in v),
                           'questions_with_verified_trace': sum(x['pass@k'] for x in v)}
                       for s, v in sorted(by_stratum.items())},
        'by_audit_status': {s: {'questions': len(v),
                                'questions_with_verified_trace': sum(x['pass@k'] for x in v)}
                            for s, v in sorted(by_status.items())},
        'per_question': per_q, 'training_eligible': False,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    verdict = lambda g: '通过' if gates[g]['pass'] else '未达标'
    lines = ['# 2c 分层扩测（50 题 × k=4，掩码解码器）', '',
             f"抽样框 `label_disposition/sampling_frame_50.jsonl`（500 题池按子类配额的 50 题、{len({v['db_id'] for v in per_q.values()})} 个 DB），"
             '同 student prompt/工具/观测/预算/判分；掩码解码器（不带 stop、不约束生成、事后 token 掩码）；',
             '预注册门槛与判读规则见 `../GOAL.md` §3–§4。**本表不导出训练数据**（`training_eligible: false`）。', '',
             '## 预注册门槛', '',
             '| 门槛 | 预注册 | 实测 | 判定 |', '|---|---:|---:|---|',
             f"| ① 掩码后动作解析率 | ≥99% | {parse_rate:.4f}（{responses - no_action}/{responses} 响应解析出第一个完整动作） | {verdict('parse_rate')} |",
             f"| ② 提交率 | ≥95% | {submit_rate:.4f}（{submitted}/{len(valid)} 条有效 episode） | {verdict('submit_rate')} |",
             f"| ③ ≥1 条 verified trace 的题比例 | ≥60% | {verified_rate:.4f}（{verified_questions}/{len(ids)} 题；verified=三闸定义预演） | {verdict('verified_question_rate')} |", '',
             '## 汇总', '',
             f"- 有效 episode {len(valid)}/{len(recs)}（提供方协议失败重跑后仍失败 {len(failures)} 条，单独列出，不计入能力证据）；",
             f"- Gold EX（episode）{verified}/{len(valid)} = {verified / len(valid):.3f}；pass@{max((v['k_valid'] for v in per_q.values()), default=0)} 口径「至少 1 条 verified」的题 {verified_questions}/{len(ids)}；",
             f"- 无动作响应 {no_action}；空可见响应属提供方协议失败，已在上面单列；",
             f"- API 调用 {summary['calls_spent']}（上限 {summary['call_budget']}，budget_exhausted={summary['budget_exhausted']}）。", '',
             '## 分层', '',
             '| 分层（valid_of_4） | 题数 | 有效 episode | verified episode | ≥1 verified 的题 |', '|---|---:|---:|---:|---:|']
    for s in STRATUM_ORDER:
        if s in summary['by_stratum']:
            d = summary['by_stratum'][s]
            lines.append(f"| {s} | {d['questions']} | {d['episodes_valid']} | {d['verified_episodes']} | {d['questions_with_verified_trace']} |")
    lines += ['', '## 逐题', '',
              '| id | DB | 分层 | per-sample | mean | 提交 | ≥1 verified | 协议失败 | 结束原因 |',
              '|---|---|---|---|---:|---:|---|---:|---|']
    for qid, v in sorted(per_q.items(), key=lambda kv: (kv[1]['db_id'], kv[0])):
        vec = [x for x in v['vector'] if x is not None]
        mean = sum(vec) / len(vec) if vec else 0.0
        lines.append(f"| {qid} | {v['db_id']} | {v['stratum']} | {v['vector']} | {mean:.2f} | "
                     f"{v['submitted']}/{v['k_valid']} | {'是' if v['pass@k'] else '否'} | {v['protocol_failure']} | {v['finish_reasons']} |")
    if failures:
        lines += ['', '## 提供方协议失败（重跑后仍失败，不计入能力证据）', '',
                  '| id | DB | k | 结束原因 | 重试次数 | 备注 |', '|---|---|---:|---|---:|---|']
        for f in failures:
            lines.append(f"| {f['question_id']} | {f['db_id']} | {f['sample_idx']} | {f['finish_reason']} | "
                         f"{f['attempt']} | {(f['reason'] or '')[:60]} |")
    lines += ['', '## 使用边界', '',
              '- 失败归因三分（GOAL §4）：协议失败 ≠ 标签冲突 ≠ SQL 能力错误；本表只做 2c 汇总，正式入闸在 2d（三道闸+审计留痕）。',
              '- 门槛③按「三闸定义预演」计：仅要求 episode 判分正确且奖励可用；2d 会追加污染检查与冲突题的 Evidence 口径。',
              '- 抽样框全部为 D2 之下 training-eligible 的题；冲突/模糊题（332 条 eval-only）不进入本池。', '']
    (OUT / 'EXPANSION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'gates': {k: v['pass'] for k, v in gates.items()},
                      'questions': len(ids), 'episodes_valid': len(valid),
                      'verified': verified, 'verified_questions': verified_questions,
                      'protocol_failures': len(failures)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
