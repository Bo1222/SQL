"""CPU-only descriptive audit of completed E1 SFT/control dev200 trajectories."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/m6/e1_alignment'


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def outcome(r):
    if r['correct']:
        return 'correct'
    return 'semantic_wrong' if r['valid'] else r.get('error_type') or 'no_answer'


def describe(rows):
    return {'n': len(rows), 'correct': sum(bool(r['correct']) for r in rows),
            'outcomes': dict(Counter(outcome(r) for r in rows)),
            'lookup_episodes': sum(r['schema_lookups'] > 0 for r in rows),
            'execute_episodes': sum(r['sql_executes'] > 0 for r in rows),
            'tool_paths': dict(Counter('>'.join([e['action'] for e in r['tool_events']]
                                              + [r['finish_reason']]) for r in rows))}


# Manual primary SQL-difference labels, not independently established causal roots.
# Keys are one-based lines in the immutable input whose hash is saved below.
MANUAL = {
    'join_table_grain': {10, 44, 86, 96, 167, 186, 196},
    'projection_aggregation_order': {38, 63, 67, 83, 84, 100, 104, 110, 114, 118, 160, 195},
    'literal_date_output_encoding': {11, 15, 50, 57, 78, 152, 181},
    'annotation_or_interpretation_needs_review': {56, 101, 138, 198},
}


def main():
    paths = [OUT / 'dev200_sft.jsonl', OUT / 'dev200_v2.jsonl',
             ROOT / 'artifacts/m6/e1_alignment/e1_alignment_sft.jsonl']
    base, v2, train = map(read, paths)
    index = {r['id']: r for r in base}
    fixed = {r['id'] for r in read(ROOT / 'artifacts/datasets/dev200.jsonl')}
    assert len(base) == len(v2) == len(index) == len(fixed) == 200
    assert set(index) == {r['id'] for r in v2} == fixed
    losses = [r for r in v2 if index[r['id']]['correct'] and not r['correct']]
    gains = [r for r in v2 if not index[r['id']]['correct'] and r['correct']]
    columns = [r for r in v2 if r.get('error_type') == 'no_such_column']
    column_tools = [r for r in columns if r['tool_events']]
    semantic_lines = {i for i, r in enumerate(v2, 1) if not r['correct'] and r['valid']}
    assert set.union(*MANUAL.values()) == semantic_lines
    assert sum(map(len, MANUAL.values())) == len(semantic_lines)
    result = {
        'source_type': 'measured existing trajectories; descriptive counts, no new model evaluation',
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths},
        'sft': describe(base), 'v2': describe(v2),
        'v2_wrong': describe([r for r in v2 if not r['correct']]),
        'v2_direct': describe([r for r in v2 if not r['tool_events']]),
        'v2_with_tools': describe([r for r in v2 if r['tool_events']]),
        'losses': describe(losses), 'gains': describe(gains),
        'column_errors': describe(columns),
        'column_errors_after_tools': {
            **describe(column_tools),
            'last_execution_error': sum(r['executions'][-1]['result']['status'] == 'error'
                                        for r in column_tools),
            'final_equals_last_executed_sql': sum(r['executions'][-1]['sql'].strip()
                                                 == r['final_sql'].strip() for r in column_tools)},
        'by_db': {db: {'sft': describe([r for r in base if r['db_id'] == db]),
                       'v2': describe([r for r in v2 if r['db_id'] == db])}
                  for db in sorted({r['db_id'] for r in v2})},
        'transitions': dict(Counter(outcome(index[r['id']]) + '->' + outcome(r) for r in v2)),
        'semantic_manual_primary_difference_counts': {k: len(v) for k, v in MANUAL.items()},
    }
    training = {}
    for source in sorted({r['trajectory_source'] for r in train}):
        rows = [r for r in train if r['trajectory_source'] == source]
        first, supervised, tokens = Counter(), Counter(), Counter()
        for r in rows:
            actions = [s for s in r['segments'] if s['kind'] == 'action']
            first[actions[0]['action']] += 1
            if any(actions[0]['loss_mask']):
                supervised[actions[0]['action']] += 1
            for s in actions:
                tokens[s['action']] += sum(s['loss_mask'])
        training[source] = {'n': len(rows), 'first_actions': dict(first),
                            'supervised_first_actions': dict(supervised),
                            'supervised_tokens_by_action': dict(tokens)}
    result['training_action_supervision'] = training
    cases = []
    for line, r in enumerate(v2, 1):
        b = index[r['id']]
        category = next((k for k, lines in MANUAL.items() if line in lines), None)
        cases.append({'line': line, 'id': r['id'], 'db_id': r['db_id'],
                      'question': r['question'], 'gold_sql': r['gold_sql'],
                      'sft_outcome': outcome(b), 'v2_outcome': outcome(r),
                      'sft_sql': b['final_sql'], 'v2_sql': r['final_sql'],
                      'sft_tools': [e['action'] for e in b['tool_events']],
                      'v2_tools': [e['action'] for e in r['tool_events']],
                      'v2_execution_feedback': [e['result'] for e in r['executions']],
                      'manual_primary_sql_difference': category})
    result['cases'] = cases
    (OUT / 'v2_error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'by_db', 'input_sha256']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
