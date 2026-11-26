"""Action-level comparison: same 10 questions, 3B student (n=32 rollouts each)
vs the GLM teacher under the same typed harness.

Two different quantities, kept apart on purpose:

  * protocol survival -- did the episode get a submission at all, and how many
    responses failed the single-action parse (`invalid_actions`).
  * executed-action quality -- of the actions that DID execute, how many came
    back as tool errors (`schema_error`, `syntax_error`).

Reads reports only; writes trace_masked/action_error_comparison.json.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/m6/glm_hard10_20261123'
TRACE = OUT / 'trace_masked'
STUDENT = ROOT / 'reports/m6/sft_first_20261120/self_sampling/rollouts32/records.jsonl'
read = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def teacher_side(ids):
    traces = {t['id']: t for t in read(TRACE / 'traces.jsonl')}
    assert set(traces) == set(ids), sorted(set(ids) ^ set(traces))
    steps = [s for t in traces.values() for s in t['steps']]
    executed = [e for t in traces.values() for e in t['tool_events']]
    sql = [e for e in executed if e['action'] == 'sql_execute']
    look = [e for e in executed if e['action'] == 'schema_lookup']
    sql_bad = [e for e in sql if e['result'].get('status') != 'ok']
    look_bad = [e for e in look if e['result'].get('status') != 'ok']
    return {
        'questions': len(traces),
        'submitted': sum(t['finish_reason'] == 'answer_submitted' for t in traces.values()),
        'finish_reasons': dict(Counter(t['finish_reason'] for t in traces.values())),
        'responses': len(steps),
        'responses_with_action': sum(s['outcome'] == 'action' or 'masked_action_text' in s for s in steps),
        'responses_no_action': sum(s['outcome'] == 'no_action' for s in steps),
        'responses_invalid_action': sum(s['outcome'] == 'invalid_action' for s in steps),
        'responses_over_budget': sum(s['outcome'] == 'action_over_budget' for s in steps),
        'executed_actions': len(executed),
        'sql_executes': len(sql), 'sql_errors': len(sql_bad),
        'sql_error_rate': round(len(sql_bad) / len(sql), 4) if sql else None,
        'sql_error_types': dict(Counter(e['result'].get('error_type') for e in sql_bad)),
        'schema_lookups': len(look), 'schema_lookup_errors': len(look_bad),
        'extra_actions_dropped': sum(t['extra_actions_dropped'] for t in traces.values()),
        'prompt_side_prose_suppression': False,
    }


def student_side(ids):
    rows = read(STUDENT)
    rows = [r for r in rows if r['id'] in ids]
    assert len({r['id'] for r in rows}) == len(ids)
    actions = [e for r in rows for e in r['tool_events']]
    sql = [e for e in actions if e['action'] == 'sql_execute']
    look = [e for e in actions if e['action'] == 'schema_lookup']
    sql_bad = [e for e in sql if e['result'].get('status') != 'ok']
    look_bad = [e for e in look if e['result'].get('status') != 'ok']
    return {
        'questions': len({r['id'] for r in rows}), 'rollouts': len(rows),
        'rollouts_per_question': len(rows) // len({r['id'] for r in rows}),
        'submitted': sum(r['finish_reason'] == 'answer_submitted' for r in rows),
        'finish_reasons': dict(Counter(r['finish_reason'] for r in rows)),
        'invalid_actions_total': sum(r['invalid_actions'] for r in rows),
        'rollouts_with_invalid_action': sum(r['invalid_actions'] > 0 for r in rows),
        'executed_actions': len(actions),
        'sql_executes': len(sql), 'sql_errors': len(sql_bad),
        'sql_error_rate': round(len(sql_bad) / len(sql), 4) if sql else None,
        'sql_error_types': dict(Counter(e['result'].get('error_type') for e in sql_bad)),
        'schema_lookups': len(look), 'schema_lookup_errors': len(look_bad),
    }


def main():
    ids = {t['id'] for t in read(TRACE / 'traces.jsonl')}
    out = {'question_ids': sorted(ids),
           'student_3b_self_sampling': student_side(ids),
           'glm_teacher_masked_trace': teacher_side(ids),
           'note': ('Student numbers are 32 sampled rollouts per question; teacher numbers are one '
                    'trajectory per question. Protocol survival and executed-action quality are '
                    'different quantities and are not comparable one-to-one.')}
    (TRACE / 'action_error_comparison.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
