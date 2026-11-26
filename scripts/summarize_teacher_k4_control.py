"""Summarize the 2a decoder control (masked vs strict, k=4, ten pilot questions).

Reading rule is pre-registered in reports/m6/teacher_pool_20261123/GOAL.md §4:
compare per-question pass@4 across arms against the within-arm half-sample
fluctuation; treat everything as a decoder/protocol control, never as a
capability ceiling. Conflict questions are reported separately throughout.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/m6/teacher_pool_20261123/k4_control'
ARMS = ('masked', 'strict')
CONFLICT = ('gold_question_projection_conflict', 'gold_evidence_conflict',
            'evidence_date_format_conflict', 'output_layout_ambiguous')
EVIDENCE_MARKERS = {  # heuristic, for the five pre-flagged conflict questions only
    'bird_train_7d760291cd5d733d8145': (re.compile(r"'Liquor'", re.I), re.compile(r"'Restaurant'", re.I)),
    'bird_train_0d0e403a1a7d3c0b127e': (re.compile(r"EmailPromotion\s*=\s*1", re.I),
                                        re.compile(r"EmailPromotion\s*=\s*0", re.I)),
    'bird_train_09fe02c588acd2b550f1': (re.compile(r"'1/1/18'", re.I), re.compile(r"'1/1/2018'", re.I)),
    'bird_train_0e6f6535a44f3a88fa09': (re.compile(r"SUM\s*\(\s*CASE", re.I), re.compile(r"GROUP\s+BY", re.I)),
    'bird_train_42106bdd27ea14b7197d': (re.compile(r"^(?!.*(?:FirstName|LastName)).*SELECT", re.I | re.S),
                                         re.compile(r"FirstName|LastName", re.I)),
}


def load(arm):
    path = OUT / arm / 'episodes.jsonl'
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r['question_id'], r['sample_idx'])] = r
    return latest


def is_provider_failure(r):
    return bool(r.get('provider_protocol_failure', r.get('infrastructure_error')
                      or r.get('finish_reason') == 'generation_error'))


def half_disagreement(vec):
    return abs(sum(vec[:2]) - sum(vec[2:])) / 2


def main():
    data = {arm: load(arm) for arm in ARMS}
    ids = sorted({k[0] for k in data['masked']} | {k[0] for k in data['strict']})
    audit = {r['id']: r['audit_status'] for r in
             map(json.loads, (ROOT / 'reports/m6/glm_hard10_20261123/audit.jsonl').read_text().splitlines())}
    per_q, noise = {}, {arm: [] for arm in ARMS}
    for qid in ids:
        row = {'question_id': qid, 'audit_status': audit.get(qid), 'arms': {}}
        for arm in ARMS:
            eps = [data[arm][(qid, k)] for k in sorted(s for i, s in data[arm] if i == qid)]
            valid = [e for e in eps if not is_provider_failure(e)]
            vec = [int(e['correct'] and e['reward_available']) for e in valid]
            row['arms'][arm] = {
                'k': len(eps), 'k_valid': len(valid), 'vector': vec,
                'mean': sum(vec) / len(vec) if vec else None,
                'pass@k': int(any(vec)), 'submitted': sum(e['submitted'] for e in valid),
                'provider_protocol_failure': sum(is_provider_failure(e) for e in eps),
                'finish_reasons': dict(Counter(e['finish_reason'] for e in valid)),
                'protocol': {key: sum((e['protocol'] or {}).get(key, 0) for e in valid)
                             for key in ('no_action_responses', 'extra_actions_dropped', 'invalid_actions',
                                         'action_over_budget', 'boundary_tokens_reencoded')},
                'api_calls': sum(e['api_calls'] for e in eps)}
            if len(vec) == 4:
                noise[arm].append(half_disagreement(vec))
        m, s = row['arms']['masked'], row['arms']['strict']
        row['delta_pass@k'] = (m['pass@k'] or 0) - (s['pass@k'] or 0)
        row['delta_mean'] = (m['mean'] or 0) - (s['mean'] or 0)
        if qid in EVIDENCE_MARKERS:
            gold_re, evidence_re = EVIDENCE_MARKERS[qid]
            for arm in ARMS:
                sqls = [e['final_sql'] or '' for (i, _), e in sorted(data[arm].items())
                        if i == qid and not is_provider_failure(e)]
                row['arms'][arm]['marker'] = {
                    'gold_marker': sum(bool(gold_re.search(s)) for s in sqls),
                    'evidence_marker': sum(bool(evidence_re.search(s)) for s in sqls), 'n': len(sqls)}
        per_q[qid] = row

    clean_ids = [q for q in ids if audit.get(q) not in CONFLICT]
    conflict_ids = [q for q in ids if audit.get(q) in CONFLICT]
    agg = {}
    for arm in ARMS:
        eps = [e for e in data[arm].values() if not is_provider_failure(e)]
        agg[arm] = {'episodes': len(eps), 'questions': len(ids),
                    'correct_episodes': sum(e['correct'] and e['reward_available'] for e in eps),
                    'submitted': sum(e['submitted'] for e in eps),
                    'provider_protocol_failure': sum(is_provider_failure(e) for e in data[arm].values()),
                    'api_calls': sum(e['api_calls'] for e in data[arm].values()),
                    'protocol': {key: sum((e['protocol'] or {}).get(key, 0) for e in eps)
                                 for key in ('no_action_responses', 'extra_actions_dropped', 'invalid_actions',
                                             'action_over_budget', 'boundary_tokens_reencoded')},
                    'half_sample_disagreement_mean': round(sum(noise[arm]) / len(noise[arm]), 3) if noise[arm] else None,
                    'pass@k_clean': sum(per_q[q]['arms'][arm]['pass@k'] for q in clean_ids),
                    'pass@k_conflict': sum(per_q[q]['arms'][arm]['pass@k'] for q in conflict_ids)}
    summary = {'scope': 'decoder control on the ten pilot questions; not a capability ceiling',
               'clean_questions': len(clean_ids), 'conflict_questions': len(conflict_ids),
               'aggregate': agg,
               'mean_abs_delta_pass@k_all': round(sum(abs(per_q[q]['delta_pass@k']) for q in ids) / len(ids), 3),
               'mean_abs_delta_pass@k_clean': round(sum(abs(per_q[q]['delta_pass@k']) for q in clean_ids) / len(clean_ids), 3),
               'noise_scale_note': 'half_sample_disagreement_mean is the mean |correct(first 2) - correct(last 2)|/2 within an arm; the pre-registered reading is |delta| within that scale => mask neutral',
               'training_eligible': False, 'per_question': per_q}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    lines = ['# 2a 解码器对照（掩码 vs 严格，k=4）', '',
             '同 10 道 pilot 题、同 prompt/工具/预算/判分，唯一差异是解码器：掩码臂不带 `stop`、服务端原样返回后按 token 掩码取第一个完整动作；',
             '严格臂带 `stop=[闭合标签]`、只在响应以动作开头时恢复被省略的闭合标签，无动作响应按学生 harness 记违规。',
             '**这是解码器/协议对照，不是能力上界**；5 道事前判冲突题的判分不构成能力证据。判读规则见 `../GOAL.md` §4。', '',
             f"干净题 {len(clean_ids)} 道、冲突题 {len(conflict_ids)} 道。", '',
             '## 汇总', '',
             '| 口径 | 掩码臂 | 严格臂 |', '|---|---:|---:|',
             f"| 提交/回答 | {agg['masked']['submitted']}/{agg['masked']['episodes']} | {agg['strict']['submitted']}/{agg['strict']['episodes']} |",
             f"| Gold EX（episode 数） | {agg['masked']['correct_episodes']}/{agg['masked']['episodes']} | {agg['strict']['correct_episodes']}/{agg['strict']['episodes']} |",
             f"| pass@4（干净 5 题） | {agg['masked']['pass@k_clean']}/5 | {agg['strict']['pass@k_clean']}/5 |",
             f"| pass@4（冲突 5 题） | {agg['masked']['pass@k_conflict']}/5 | {agg['strict']['pass@k_conflict']}/5 |",
             f"| 无动作响应 | {agg['masked']['protocol']['no_action_responses']} | — |",
             f"| 多动作被丢弃 | {agg['masked']['protocol']['extra_actions_dropped']} | — |",
             f"| 违规/解析失败 | {agg['masked']['protocol']['invalid_actions']} | {agg['strict']['protocol']['invalid_actions']} |",
             f"| 掩码边界重编码 | {agg['masked']['protocol']['boundary_tokens_reencoded']} | — |",
             f"| 提供方协议失败（重跑后仍失败） | {agg['masked']['provider_protocol_failure']} | {agg['strict']['provider_protocol_failure']} |",
             f"| API 调用 | {agg['masked']['api_calls']} | {agg['strict']['api_calls']} |",
             f"| 同臂半样本不一致均值 | {agg['masked']['half_sample_disagreement_mean']} | {agg['strict']['half_sample_disagreement_mean']} |", '',
             f"逐题 |pass@4(掩码) − pass@4(严格)| 均值：全部 {summary['mean_abs_delta_pass@k_all']}，干净题 {summary['mean_abs_delta_pass@k_clean']}。",
             '判读：若该差落在同臂半样本不一致的尺度内，则判「掩码不损害教师规划」；超出则逐题归因。', '',
             '## 逐题', '',
             '| DB / id | 审阅 | 掩码 per-sample | 掩码 mean | 严格 per-sample | 严格 mean | Δpass@4 | 严格结束原因 |',
             '|---|---|---|---:|---|---:|---:|---|']
    for qid, row in per_q.items():
        m, s = row['arms']['masked'], row['arms']['strict']
        lines.append(f"| {qid} | {row['audit_status']} | {m['vector']} | {m['mean']:.2f} | {s['vector']} | {s['mean']:.2f} | "
                     f"{row['delta_pass@k']:+d} | {s['finish_reasons']} |")
    lines += ['', '## 冲突题的 Evidence/Gold 走向（启发式标记，仅这 5 题）', '',
              '| id | 臂 | 命中 Gold 标记 | 命中 Evidence 标记 | 可判定轨迹数 |', '|---|---|---:|---:|---:|']
    for qid in conflict_ids:
        if qid not in EVIDENCE_MARKERS:
            continue
        for arm in ARMS:
            mk = per_q[qid]['arms'][arm].get('marker')
            if mk:
                lines.append(f"| {qid} | {arm} | {mk['gold_marker']} | {mk['evidence_marker']} | {mk['n']} |")
    lines += ['', '标记含义按题固定（Liquor/Restaurant、EmailPromotion=1/0、"1/1/18"/"1/1/2018"、透视/GROUP BY、是否投影姓名）；',
              '两条标记都未命中或同时命中时该轨迹不计入任何一类。启发式只用于观察方向，不替代逐条复核。', '',
              '## 使用边界', '',
              '`training_eligible: false`；本运行不导出训练数据、不训练模型。',
              '严格臂的「无动作→违规」是学生 harness 的既有语义，不是 SQL 能力证据；两臂的 Gold EX 都只在干净 5 题上可解释。',
              '证据：k4_control/{masked,strict}/episodes.jsonl、api_responses.jsonl、run_summary.json、run.log。']
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == '__main__':
    main()
