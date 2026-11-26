"""2d: intake gates for 2c teacher traces (candidate list + audit trail only).

GOAL.md §3 row 2d pre-registers three gates per trace:
  ① protocol clean  — submitted, every response parsed to a complete action, no
     invalid actions, no infrastructure/generation error, and the runtime
     fingerprint equals the student contract (same-condition requirement).
  ② execution verified — correct under the project scorer (`bird_set`,
     set(gold)==set(pred)) and reward_available (final SQL executed locally).
  ③ contamination clean — Gold SQL never appears in teacher-visible content
     (initial prompt + API request messages + tool observations), question id is
     outside seen-11 / held-out-20 / dev200 / test1534 / val, and db_id is not a
     held-out-20 database. D2 also requires a train_eligible disposition.

Outputs (reports/m6/teacher_pool_20261123/intake/):
  intake_report.jsonl  one record per latest (id, sample_idx): gate results,
                       reasons, and the audit material (final/gold SQL, checks).
  candidates.jsonl     pointer manifest of traces that pass all gates. Contains
                       no context/target content — not a training file.
  summary.json         aggregates (per gate / stratum / question).
  INTAKE.md            human-readable report.

training_eligible stays false (D3: third-party teacher output; licensing is a
user decision).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / 'reports/m6/teacher_pool_20261123/expansion_50/traces.jsonl'
PAIRS = ROOT / 'reports/m6/teacher_pool_20261123/expansion_50/sft_pairs.jsonl'
SPLIT = ROOT / 'reports/m6/sft_first_20261120/self_sampling/split_manifest.json'
DEV200 = ROOT / 'data/processed/dev_200.jsonl'
TEST1534 = ROOT / 'data/processed/bird_agent/test.jsonl'
VAL = ROOT / 'data/processed/bird_agent/val.jsonl'
OUT = ROOT / 'reports/m6/teacher_pool_20261123/intake'
EXPECTED_FP = '13a42bb9f55610a831adf18350d7625ac62fa4b5bc21fd7e3b919c7e75e985f9'
D2_OK = ('train_eligible', 'train_eligible_by_rule')
STRATUM_ORDER = ('1of4', '2of4', '3of4', 'all4')
STRATUM_LABEL = {1: '1of4', 2: '2of4', 3: '3of4', 4: 'all4'}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def norm(s):
    return ' '.join(str(s).lower().split())


def visible_text(trace):
    """Everything the teacher could see: prompt, request messages, tool results."""
    parts = [trace.get('prompt') or '']
    for call in trace.get('api_calls') or []:
        for msg in (call.get('request') or {}).get('messages') or []:
            parts.append(msg.get('content') or '')
    for event in trace.get('tool_events') or []:
        parts.append(json.dumps(event, ensure_ascii=False))
    return norm('\n'.join(parts))


def main():
    traces = read_jsonl(TRACES)
    latest = {}
    for t in traces:
        latest[(t['id'], t.get('sample_idx', 0))] = t
    pair_count = Counter((p['id'], p.get('sample_idx', 0)) for p in read_jsonl(PAIRS))

    split = json.loads(SPLIT.read_text())
    seen = {t['id'] for t in split['seen']}
    heldout_ids = {t['id'] for t in split['heldout']}
    heldout_dbs = {t['db_id'] for t in split['heldout']}
    eval_ids = {}
    for name, path in (('dev200', DEV200), ('test1534', TEST1534), ('val590', VAL)):
        eval_ids[name] = {str(r['id']) for r in read_jsonl(path)}

    OUT.mkdir(parents=True, exist_ok=True)
    report, candidates = [], []
    for (qid, k), t in sorted(latest.items()):
        reasons = []
        protocol_clean = (t.get('finish_reason') == 'answer_submitted'
                          and (t.get('no_action_responses') or 0) == 0
                          and (t.get('invalid_actions') or 0) == 0
                          and not t.get('infrastructure_error')
                          and not t.get('generation_error'))
        if not protocol_clean:
            detail = []
            if t.get('finish_reason') != 'answer_submitted':
                detail.append(str(t.get('finish_reason')))
            else:
                if (t.get('no_action_responses') or 0) > 0:
                    detail.append(f"no_action_responses={t['no_action_responses']}")
                if (t.get('invalid_actions') or 0) > 0:
                    detail.append(f"invalid_actions={t['invalid_actions']}")
                if t.get('infrastructure_error'):
                    detail.append('infrastructure_error')
                if t.get('generation_error'):
                    detail.append('generation_error')
            reasons.append('protocol:' + '+'.join(detail))
        fp_ok = t.get('runtime_fingerprint') == EXPECTED_FP
        if not fp_ok:
            reasons.append('fingerprint_mismatch')
        execution_verified = bool(t.get('correct') and t.get('reward_available') and t.get('final_sql'))
        if not execution_verified:
            reasons.append('not_verified')
        gold_leak = norm(t.get('gold_sql') or '') in visible_text(t) if t.get('gold_sql') else False
        checks = {
            'gold_leak': gold_leak,
            'seen_hit': qid in seen,
            'heldout_id_hit': qid in heldout_ids,
            'heldout_db_hit': t.get('db_id') in heldout_dbs,
            'dev200_hit': str(qid) in eval_ids['dev200'],
            'test1534_hit': str(qid) in eval_ids['test1534'],
            'val_hit': str(qid) in eval_ids['val590'],
        }
        for name, hit in checks.items():
            if hit:
                reasons.append('contamination:' + name)
        d2_ok = str(t.get('audit_status', '')).startswith('train_eligible')
        if not d2_ok:
            reasons.append('d2_not_eligible:' + str(t.get('audit_status')))
        candidate = (protocol_clean and execution_verified and fp_ok
                     and not any(checks.values()) and d2_ok)
        rec = {
            'id': qid, 'db_id': t.get('db_id'), 'sample_idx': k, 'attempt': t.get('attempt'),
            'stratum': STRATUM_LABEL.get(t.get('stratum_of_4'), str(t.get('stratum_of_4'))),
            'audit_status': t.get('audit_status'),
            'finish_reason': t.get('finish_reason'), 'rounds': t.get('rounds'),
            'correct': bool(t.get('correct')), 'reward_available': bool(t.get('reward_available')),
            'final_sql': t.get('final_sql'), 'gold_sql': t.get('gold_sql'),
            'pair_count': pair_count.get((qid, k), 0),
            'gates': {'protocol_clean': protocol_clean, 'execution_verified': execution_verified,
                      'contamination_clean': not any(checks.values()), 'fingerprint_ok': fp_ok,
                      'd2_eligible': d2_ok},
            'checks': checks, 'reasons': reasons, 'candidate': candidate,
            'runtime_contract': t.get('runtime_contract'),
            'runtime_fingerprint': t.get('runtime_fingerprint'),
            'training_eligible': False,
            'training_note': 'third-party teacher output; licensing decision pending (D3)',
        }
        report.append(rec)
        if candidate:
            import hashlib
            candidates.append({
                'id': qid, 'db_id': t.get('db_id'), 'sample_idx': k,
                'stratum': rec['stratum'], 'audit_status': t.get('audit_status'),
                'pair_count': rec['pair_count'],
                'final_sql_sha256': hashlib.sha256((t.get('final_sql') or '').encode()).hexdigest(),
                'runtime_fingerprint': t.get('runtime_fingerprint'),
                'training_eligible': False,
                'training_note': 'pointer only; no context/target content in this file',
            })
    (OUT / 'intake_report.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in report))
    (OUT / 'candidates.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in candidates))

    by_stratum = defaultdict(lambda: {'traces': 0, 'candidates': 0})
    per_q = defaultdict(lambda: {'traces': 0, 'candidates': 0, 'stratum': None, 'db_id': None,
                                 'audit_status': None})
    for r in report:
        s = by_stratum[r['stratum']]
        s['traces'] += 1
        s['candidates'] += int(r['candidate'])
        q = per_q[r['id']]
        q['traces'] += 1
        q['candidates'] += int(r['candidate'])
        q.update(stratum=r['stratum'], db_id=r['db_id'], audit_status=r['audit_status'])
    reason_counts = Counter(x.split(':')[0] if ':' not in x else x.split(':')[0] + ':' + x.split(':', 1)[1]
                            for r in report for x in r['reasons'])
    summary = {
        'scope': '2d intake; candidate list + audit trail only, no training files (D3)',
        'questions': len(per_q), 'traces_latest': len(report),
        'protocol_clean': sum(r['gates']['protocol_clean'] for r in report),
        'execution_verified': sum(r['gates']['execution_verified'] for r in report),
        'contamination_clean': sum(r['gates']['contamination_clean'] for r in report),
        'fingerprint_ok': sum(r['gates']['fingerprint_ok'] for r in report),
        'd2_eligible': sum(r['gates']['d2_eligible'] for r in report),
        'candidates': len(candidates),
        'questions_with_candidate': sum(1 for q in per_q.values() if q['candidates']),
        'candidates_per_question': dict(Counter(q['candidates'] for q in per_q.values())),
        'reason_counts': dict(reason_counts),
        'by_stratum': {s: by_stratum[s] for s in STRATUM_ORDER if s in by_stratum},
        'per_question': dict(sorted(per_q.items())),
        'training_eligible': False,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    lines = ['# 2d 入闸（三道）— 2c 轨迹候选清单', '',
             f"输入：`expansion_50/traces.jsonl` 的 latest-per-(id, sample_idx) 共 {len(report)} 条（50 题）。",
             '闸门定义预注册于 `../GOAL.md` §3 row 2d；判读规则 §4。**只产候选清单 + 审计留痕，不导出训练文件**（D3）。', '',
             '## 三道闸', '',
             '| 闸 | 定义 | 通过 | 未通过 |', '|---|---|---:|---:|',
             f"| ① 协议干净 | 提交 ∧ 无动作响应=0 ∧ 违规动作=0 ∧ 无基础设施/生成错误 ∧ 运行时指纹=student 契约 | {summary['protocol_clean']}/{len(report)} | {len(report)-summary['protocol_clean']} |",
             f"| ② 执行验证 | `bird_set` 判分正确 ∧ reward_available ∧ 有 final_sql | {summary['execution_verified']}/{len(report)} | {len(report)-summary['execution_verified']} |",
             f"| ③ 污染干净 | Gold 不出现在教师可见内容 ∧ 题号 ∉ seen-11/held-out-20/dev200/test1534/val590 ∧ DB ∉ held-out-20 | {summary['contamination_clean']}/{len(report)} | {len(report)-summary['contamination_clean']} |",
             f"| D2 资格 | disposition ∈ train_eligible（含 by_rule） | {summary['d2_eligible']}/{len(report)} | {len(report)-summary['d2_eligible']} |",
             '',
             f"**候选 {len(candidates)} 条，覆盖 {summary['questions_with_candidate']}/50 题。**", '',
             '## 分层', '',
             '| 分层 | 轨迹 | 候选 |', '|---|---:|---:|']
    for s in STRATUM_ORDER:
        if s in summary['by_stratum']:
            d = summary['by_stratum'][s]
            lines.append(f"| {s} | {d['traces']} | {d['candidates']} |")
    lines += ['', '## 未通过清单', '']
    bad = [r for r in report if not r['candidate']]
    if bad:
        lines += ['| id | DB | k | 原因 |', '|---|---|---|---|']
        for r in bad:
            lines.append(f"| {r['id']} | {r['db_id']} | {r['sample_idx']} | {', '.join(r['reasons'])} |")
    else:
        lines.append('（无）')
    lines += ['', '## 使用边界', '',
              '- D3：`training_eligible: false`。`candidates.jsonl` 只是指针清单（id/sample_idx/hash），'
              '不含 context/target 内容，不是训练文件；是否进入 SFT 混合（步骤 3）由用户对第三方输出的授权决定。',
              '- 候选同题去重（同题 ≤N 条）在步骤 3 混合时决定；本报告只给分布。',
              '- 闸②的正确性是采集时由学生同款 `bird_set` 判分器在本地 SQLite 执行后给出；'
              '本题池全部为 label_clean（D2），不涉及冲突题的 Evidence 口径。', '']
    (OUT / 'INTAKE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'traces': len(report), 'candidates': len(candidates),
                      'questions_with_candidate': summary['questions_with_candidate'],
                      'protocol_clean': summary['protocol_clean'],
                      'execution_verified': summary['execution_verified'],
                      'contamination_clean': summary['contamination_clean'],
                      'reasons': summary['reason_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
