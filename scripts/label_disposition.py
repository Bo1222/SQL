"""2b: turn the Step-0 label triage into a per-question disposition table (CPU only).

Policy (user-approved 2026-11-23, recorded in reports/m6/teacher_pool_20261123/GOAL.md):
  * unrepairable label defects are excluded from training and from the RL reward,
    but stay in the evaluation reporting, listed separately;
  * repairable cases are *proposed*, never auto-applied (original labels stay
    untouched, `original_label_modified: false`);
  * teacher-trace generation pool = `label_clean` questions only.

Two ambiguity codes are downgraded by explicit rule (both verifiable):
  * `order_mismatch`      -> our scoring contract is bird_set, i.e.
                             set(gold) == set(pred) (tools/sql_worker.py), which
                             ignores row order entirely;
  * `slow_gold_query`     -> all 18 were re-checked and execute within 30 s
                             (step0_step1/STEP0_STEP1_SUMMARY.md).

No model, no GPU, no API. Writes no training data.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/m6/sft_first_20261120/step0_step1/label_audit.jsonl'
SUBCLASSES = ROOT / 'reports/m6/sft_first_20261120/step0_step1/sft_allzero_subclasses.jsonl'
SPLIT = ROOT / 'reports/m6/sft_first_20261120/self_sampling/split_manifest.json'
OUT = ROOT / 'reports/m6/teacher_pool_20261123/label_disposition'

FRAME_TARGET = 50
FRAME_POOL = ('semantic_valid_but_wrong',)   # the teacher-trace target pool
DOWNGRADE = {
    'order_mismatch': 'bird_set_scoring_is_order_insensitive (tools/sql_worker.py: correct = set(gold)==set(pred))',
    'slow_gold_query': 're_verified_executable_within_30s (step0_step1/STEP0_STEP1_SUMMARY.md)',
}
REPAIR_ROUTES = {
    'evidence_filter_value_not_in_gold': 'human_judgment: amend Evidence value or Gold predicate',
    'evidence_filter_column_not_in_gold': 'human_judgment: amend Evidence column or Gold predicate',
    'evidence_literal_unused': 'human_judgment: Evidence literal may be a distractor',
    'evidence_filter_column_not_in_schema': 'human_rewrite: Evidence references a column absent from schema',
    'evidence_column_not_in_schema': 'human_rewrite: Evidence references a column absent from schema',
    'evidence_filter_unknown_table': 'human_rewrite: Evidence references an unknown table',
    'evidence_table_unknown': 'human_rewrite: Evidence references an unknown table',
    'evidence_unresolved_ref': 'human_rewrite: Evidence reference cannot be resolved',
    'aggregate_mismatch': 'human_judgment: Gold aggregate does not match the question wording',
    'negation_mismatch': 'human_judgment: negation scope differs between question and Gold',
    'question_literal_missing_from_gold': 'human_judgment: question literal absent from Gold',
    'gold_literal_unmentioned': 'human_judgment: Gold literal not mentioned in question/Evidence',
    'order_mismatch': 'none_under_bird_set_scoring',
    'slow_gold_query': 'none_after_recheck',
}
read = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
write = lambda p, obj: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
codes = lambda entries: [e['code'] if isinstance(e, dict) else e for e in entries]


def dispose(entry, subclass):
    status = entry['label']['status']
    ec, am = codes(entry['label']['evidence_conflicts']), codes(entry['label']['ambiguities'])
    all_codes = ec + am
    row = {'question_id': entry['question_id'], 'db_id': entry['db_id'], 'category': entry['category'],
           'semantic_subclass': entry.get('semantic_subclass'), 'schema_subclass': entry.get('schema_subclass'),
           'audit_status': status, 'evidence_conflict_codes': ec, 'ambiguity_codes': am,
           'valid_of_4': subclass['valid'], 'valid_but_wrong_of_4': subclass['valid_but_wrong'],
           'original_label_modified': False}
    if status == 'dead':
        row.update(disposition='dead', rule='gold_not_executable', training_eligible=False,
                   rl_reward_eligible=False, eval_reporting='listed_separately')
        return row
    if status == 'label_clean':
        row.update(disposition='train_eligible', rule=None, training_eligible=True,
                   rl_reward_eligible=True, eval_reporting='included')
        return row
    downgradable = [c for c in all_codes if c in DOWNGRADE]
    blocking = [c for c in all_codes if c not in DOWNGRADE]
    if not ec and not blocking:
        row.update(disposition='train_eligible_by_rule',
                   rule='; '.join(f'{c}: {DOWNGRADE[c]}' for c in sorted(set(downgradable))),
                   training_eligible=True, rl_reward_eligible=True, eval_reporting='included')
        return row
    routes = sorted({REPAIR_ROUTES.get(c, 'human_judgment: unclassified code') for c in blocking})
    row.update(disposition='eval_only_conflict' if ec else 'eval_only_ambiguous',
               rule='D2: excluded from training and RL reward, kept in evaluation, listed separately',
               repair_routes=routes, training_eligible=False, rl_reward_eligible=False,
               eval_reporting='listed_separately')
    return row


def build_frame(rows, seen_ids, heldout_dbs):
    pool = [r for r in rows if r['disposition'].startswith('train_eligible')
            and r['category'] in FRAME_POOL and r['question_id'] not in seen_ids
            and r['db_id'] not in heldout_dbs]
    by_subclass = defaultdict(list)
    for r in pool:
        by_subclass[r['semantic_subclass']].append(r)
    total = len(pool)
    quotas, rema = {}, []
    for s, items in sorted(by_subclass.items()):
        exact = FRAME_TARGET * len(items) / total
        quotas[s] = int(exact)
        rema.append((exact - int(exact), s))
    for _, s in sorted(rema, reverse=True)[:FRAME_TARGET - sum(quotas.values())]:
        quotas[s] += 1
    picked = []
    for s, quota in sorted(quotas.items()):
        items = sorted(by_subclass[s], key=lambda r: (r['db_id'], r['question_id']))
        by_db = defaultdict(list)
        for r in items:
            by_db[r['db_id']].append(r)
        dbs = sorted(by_db)
        i, chosen = 0, []
        while len(chosen) < quota:
            progressed = False
            for db in dbs:
                if len(chosen) >= quota:
                    break
                if i < len(by_db[db]):
                    chosen.append(by_db[db][i]); progressed = True
            if not progressed:
                break
            i += 1
        for r in chosen:
            picked.append({**{k: r[k] for k in ('question_id', 'db_id', 'category', 'semantic_subclass',
                                                'valid_of_4', 'valid_but_wrong_of_4')},
                           'stratum': s, 'disposition': r['disposition']})
    return pool, picked, quotas


def main():
    entries = read(AUDIT)
    subclasses = {r['question_id']: r for r in read(SUBCLASSES)}
    assert len(entries) == 1815 and len(subclasses) == 1815
    for e in entries:
        assert subclasses[e['question_id']]['label_status'] == e['label']['status']
    rows = [dispose(e, subclasses[e['question_id']]) for e in entries]
    assert len({r['question_id'] for r in rows}) == 1815

    split = json.loads(SPLIT.read_text())
    seen_ids = {t['id'] for t in split['seen']}
    heldout_dbs = {t['db_id'] for t in split['heldout']}
    pool, frame, quotas = build_frame(rows, seen_ids, heldout_dbs)
    assert len(frame) == FRAME_TARGET, len(frame)
    assert not ({f['question_id'] for f in frame} & seen_ids)
    assert not ({f['db_id'] for f in frame} & heldout_dbs)
    assert len({f['question_id'] for f in frame}) == FRAME_TARGET

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'label_disposition.jsonl').open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (OUT / 'sampling_frame_50.jsonl').open('w') as fh:
        for r in frame:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    counts = Counter(r['disposition'] for r in rows)
    by_cat = {c: dict(Counter(r['disposition'] for r in rows if r['category'] == c))
              for c in sorted({r['category'] for r in rows})}
    code_counts = Counter(c for r in rows for c in r['evidence_conflict_codes'] + r['ambiguity_codes'])
    routes = Counter(x for r in rows for x in r.get('repair_routes', []))
    summary = {
        'n': len(rows), 'dispositions': dict(counts), 'by_category': by_cat,
        'audit_status_unchanged': dict(Counter(r['audit_status'] for r in rows)),
        'code_counts': dict(code_counts.most_common()), 'repair_routes': dict(routes.most_common()),
        'training_eligible': sum(r['training_eligible'] for r in rows),
        'rl_reward_eligible': sum(r['rl_reward_eligible'] for r in rows),
        'eval_only': sum(r['eval_reporting'] == 'listed_separately' for r in rows),
        'original_label_modified': 0,
        'policy': 'GOAL.md D2: unrepairable label defects leave training and the RL reward, stay in evaluation listed separately; repairs are proposed, never auto-applied.',
        'frame': {'target': FRAME_TARGET, 'pool': len(pool), 'pool_definition':
                  'training-eligible AND category in semantic_valid_but_wrong AND not in seen-11 AND db not a heldout-20 db',
                  'quotas': quotas, 'dbs': len({f['db_id'] for f in frame}),
                  'picked': len(frame), 'excluded_seen_ids': sorted(seen_ids), 'excluded_heldout_dbs': sorted(heldout_dbs)},
        'training_eligible_export': False, 'no_model_no_api': True,
    }
    write(OUT / 'disposition_summary.json', summary)

    lines = ['# 2b 标签处置表（D2 政策）', '',
             'CPU-only；不改任何原始标签（`original_label_modified: false`）。队列与政策见 `../GOAL.md` §2。', '',
             '## 处置分布', '',
             '| disposition | 题数 | 训练 | RL 奖励 | 评测报告 |', '|---|---:|---|---|---|',
             f"| train_eligible | {counts['train_eligible']} | 是 | 是 | 计入 |",
             f"| train_eligible_by_rule | {counts.get('train_eligible_by_rule', 0)} | 是 | 是 | 计入（带规则注记） |",
             f"| eval_only_conflict | {counts.get('eval_only_conflict', 0)} | 否 | 否 | 单独列 |",
             f"| eval_only_ambiguous | {counts.get('eval_only_ambiguous', 0)} | 否 | 否 | 单独列 |",
             f"| dead | {counts.get('dead', 0)} | 否 | 否 | 单独列 |", '',
             '## 降级规则（可复核）', '',
             '- `order_mismatch` → 不构成障碍：判分契约是 `bird_set`，实现为 `set(gold) == set(pred)`（`tools/sql_worker.py`），行序完全不参与比较。',
             '- `slow_gold_query` → 初筛假阳性：18 题已复核，Gold 均可在 30 s 内执行（`step0_step1/STEP0_STEP1_SUMMARY.md`）。', '',
             '## 不可自动修复的处置', '',
             '本批没有任何可机械修复的冲突：全部为需要人工判定的语义分歧（改 Evidence 还是改 Gold）。',
             '所以 D2 之下它们一律 `eval_only`（训练与 RL 奖励都排除、评测报告里单独列），',
             '并附 `repair_routes` 供人工逐条决定；未获逐条批准前不修改标签。', '',
             '| code | 题数 | 处置路线 |', '|---|---:|---|']
    for code, n in code_counts.most_common():
        lines.append(f"| {code} | {n} | {REPAIR_ROUTES.get(code, 'human_judgment')} |")
    lines += ['', '## 2c 抽样框（50 题）', '',
              f"池：`train_eligible ∧ category=semantic_valid_but_wrong ∧ 非 seen-11 ∧ 非 heldout-20 的 DB` = **{len(pool)}** 题；",
              f"按 semantic_subclass 比例分配（{', '.join(f'{k}:{v}' for k, v in sorted(quotas.items()))}），",
              f"层内按 (db_id, question_id) 轮转取题以摊开 DB，得 **{len(frame)}** 题、覆盖 **{len({f['db_id'] for f in frame})}** 个 DB。",
              '排除项：seen-11 的题号；held-out 20 所用 DB（保持 DB-unseen 评测性质）。', '',
              '证据：label_disposition.jsonl、sampling_frame_50.jsonl、disposition_summary.json；',
              '输入：step0_step1/label_audit.jsonl、sft_allzero_subclasses.jsonl、self_sampling/split_manifest.json。']
    (OUT / 'DISPOSITION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'dispositions': dict(counts), 'frame': summary['frame']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
