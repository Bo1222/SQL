"""2a: k-sample decoder control, masked vs strict, on the ten pilot questions.

Purpose (pre-registered in reports/m6/teacher_pool_20261123/GOAL.md §3-4):
separate "the mask changes teacher planning" from "temperature-0.7 sampling
noise", which the single-sample pilot could not. This is a decoder-control
experiment, not a training-pool run: the ten questions stay as they were, five
of them label-conflicted, and every record is training_eligible=False.

Both arms share prompt, catalog, tools, observation encoder, budgets, student
tokenizer and scoring; the only difference is the decoder:
  * masked  -- request carries no `stop`; the provider response is taken
               verbatim; afterwards the first complete action block is kept by
               token mask and prose is dropped (agent/teacher_trace.py). A
               response without a complete action is re-sent (<=3) without
               consuming a round or an invalid-action slot.
  * strict  -- the archived student-time decoder: request carries
               stop=[closing tags]; the provider-omitted closing tag is
               restored only when the response starts with the action tag; a
               response without a parseable action consumes a round and an
               invalid-action slot.

Resume-safe: completed (arm, question, sample) triples are skipped, so an
interrupted run continues. Provider protocol failures are re-tried once in a
second pass and, if they fail again, are recorded as protocol failures -- never
silently dropped and never scored as capability evidence.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime_contract import CONTRACT_ID, FINGERPRINT, config, load_store, prompt
from agent.sql_episode import episode_to_record, run_episode
from agent.teacher_trace import run_teacher_episode, trace_record
from eval.harness_audit import AuditTokenizer
from reward.typed_rl_reward import compute_score_with_details
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

OUT = ROOT / 'reports/m6/teacher_pool_20261123/k4_control'
SOURCE = ROOT / 'reports/m6/sft_first_20261120/self_sampling/tasks_11_seen.jsonl'
ENDPOINT = 'https://tecoapi.thuwayteczz.com/v1/chat/completions'
MODEL = 'WAIC-GLM-5.3-Flash'
TOKENIZER = ROOT / 'outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model'
MAX_TOKENS = 4096
STOP = ['</schema_lookup>', '</sql_execute>', '</answer>']
ARMS = ('masked', 'strict')
CALL_BUDGET = 400  # pre-registered ceiling for 2a


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def tasks():
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    return [{k: r[k] for k in ('id', 'db_id', 'db_path', 'question', 'evidence', 'gold_sql',
                               'metadata_path', 'split')}
            for r in rows if r['id'] != 'bird_train_e2562b2e5c5a16d031be']


def restore_strict(text, finish_reason):
    """Pilot-compatible restore: only for responses that start with the tag."""
    if finish_reason != 'stop':
        return text, False
    for tag in ('schema_lookup', 'sql_execute', 'answer'):
        if text.lstrip().startswith('<' + tag + '>') and '</' + tag + '>' not in text:
            return text + '</' + tag + '>', True
    return text, False


def make_generate(arm, key, tok, calls, log, lock, meta):
    def generate(ids, budget):
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': tok.decode(ids)}],
                'temperature': 0.7, 'top_p': 1.0, 'max_tokens': MAX_TOKENS}
        if arm == 'strict':
            body['stop'] = STOP
        for attempt in range(2):
            request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                             headers={'Authorization': 'Bearer ' + key,
                                                      'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    obj = json.loads(response.read())
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt:
                    raise RuntimeError('teacher HTTP status ' + str(e.code)) from None
                time.sleep(2)
            except (urllib.error.URLError, TimeoutError):
                if attempt:
                    raise RuntimeError('teacher transport unavailable') from None
        choice = obj['choices'][0]
        message = choice.get('message', {})
        content = message.get('content') or ''
        restored = False
        if arm == 'strict':
            content, restored = restore_strict(content, choice.get('finish_reason'))
        record = {'request': body, 'model_returned': obj.get('model'), 'request_id': obj.get('id'),
                  'usage': obj.get('usage'), 'finish_reason': choice.get('finish_reason'),
                  'content': content, 'raw_content': message.get('content') or '',
                  'stop_delimiter_restored': restored,
                  'reasoning_characters': len(message.get('reasoning_content') or ''),
                  'attempts': attempt + 1, 'visible_student_tokens': len(tok.encode(content)),
                  'budget': budget, 'empty_visible_action': not content.strip()}
        calls.append(record)
        with lock:
            with log.open('a') as fh:
                fh.write(json.dumps({**meta, **record}, ensure_ascii=False) + '\n')
        if arm == 'masked':
            return {'content': content, 'finish_reason': choice.get('finish_reason'),
                    'usage': obj.get('usage'), 'request_id': obj.get('id'),
                    'model_returned': obj.get('model'), 'attempts': attempt + 1}
        return tok.encode(content)
    return generate


def episode(arm, row, tok, generate):
    store = load_store(ROOT / 'artifacts/catalog', row['db_id'], row['db_path'])
    initial = prompt(row, store)
    assert row['gold_sql'] not in initial and 'CREATE TABLE' not in initial
    lookup = SchemaLookupTool(store, token_budget=1024, count_tokens=lambda s: len(tok.encode(s)),
                              retrieval_version='E1.2')
    execute = SQLExecuteTool(row['db_path'], timeout=10, preview_rows=3, failure_policy='typed-v1')
    if arm == 'masked':
        state = run_teacher_episode(row, tok, generate, lookup, execute, config(), prompt=initial)
        record = trace_record(state, row)
    else:
        state = run_episode(row, tok, generate, lookup, execute, config(), prompt=initial)
        record = episode_to_record(state, row)
    scored = compute_score_with_details('bird',
        '<answer>' + state['final_sql'] + '</answer>' if state['final_sql'] else '',
        row['gold_sql'], {'db_path': row['db_path'], 'score_mode': 'bird_set'})
    invalid = record.get('invalid_actions')
    if invalid is None:
        invalid = sum(1 for s in record.get('steps', []) if s.get('outcome') in ('invalid_action', 'parse_failure'))
    protocol = {'no_action_responses': record.get('no_action_responses', 0),
                'extra_actions_dropped': record.get('extra_actions_dropped', 0),
                'invalid_actions': invalid,
                'action_over_budget': sum(1 for s in record.get('steps', [])
                                          if s.get('outcome') == 'action_over_budget'),
                'boundary_tokens_reencoded': sum(1 for s in record.get('steps', [])
                                                 if (s.get('mask') or {}).get('boundary_tokens_reencoded')),
                'masked_action_tokens': sum((s.get('mask') or {}).get('kept_tokens', 0)
                                            for s in record.get('steps', [])),
                'prose_tokens_dropped': sum((s.get('mask') or {}).get('dropped_tokens', 0)
                                            for s in record.get('steps', []))}
    return {'arm': arm, 'question_id': row['id'], 'db_id': row['db_id'],
            'correct': scored['correct'], 'score': scored,
            'reward_available': scored['reward_available'] and not record.get('infrastructure_error', False),
            'finish_reason': record.get('finish_reason'), 'rounds': record.get('rounds'),
            'tool_calls': record.get('tool_calls'), 'infrastructure_error': record.get('infrastructure_error', False),
            'submitted': record.get('finish_reason') == 'answer_submitted',
            'final_sql': state['final_sql'], 'protocol': protocol,
            'record': record, 'model': MODEL, 'runtime_contract': CONTRACT_ID,
            'runtime_fingerprint': FINGERPRINT, 'training_eligible': False,
            'diagnostic_only': True,
            'training_note': 'decoder-control run; not a training pool'}


def main(only, samples, budget, workers):
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / '.env').get('GLM_API_KEY')
    if not key:
        raise RuntimeError('GLM_API_KEY missing')
    rows = [r for r in tasks() if not only or r['id'] in only]
    tok = AuditTokenizer(TOKENIZER)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {arm: (OUT / arm / 'episodes.jsonl', OUT / arm / 'api_responses.jsonl') for arm in ARMS}
    for arm in ARMS:
        (OUT / arm).mkdir(exist_ok=True)
    lock = threading.Lock()
    done, latest_seen = set(), {}
    for arm in ARMS:
        if paths[arm][0].exists():
            for line in paths[arm][0].read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    latest_seen[(r['arm'], r['question_id'], r['sample_idx'])] = r
    # a triple counts as done only if its latest record is not a protocol failure,
    # so an interrupted run retries its failures instead of freezing them in
    def is_failure(r):
        return bool(r.get('provider_protocol_failure', r.get('infrastructure_error')
                          or r.get('finish_reason') == 'generation_error'))
    done = {k for k, r in latest_seen.items() if not is_failure(r)}
    spent = sum(1 for arm in ARMS if paths[arm][1].exists()
                for _ in paths[arm][1].read_text().splitlines())
    print(json.dumps({'resumed': len(done), 'calls_so_far': spent, 'budget': budget}), flush=True)

    def run_one(arm, row, idx, attempt):
        nonlocal spent
        with lock:
            if spent >= budget:
                raise RuntimeError('pre-registered call budget exhausted')
        calls = []
        meta = {'arm': arm, 'question_id': row['id'], 'sample_idx': idx, 'attempt': attempt}
        generate = make_generate(arm, key, tok, calls, paths[arm][1], lock, meta)
        try:
            record = episode(arm, row, tok, generate)
        except RuntimeError as exc:
            record = {'arm': arm, 'question_id': row['id'], 'db_id': row['db_id'], 'correct': False,
                      'reward_available': False, 'infrastructure_error': True,
                      'finish_reason': 'generation_error', 'final_sql': None,
                      'infrastructure_reason': str(exc), 'protocol': {}, 'record': None,
                      'training_eligible': False, 'diagnostic_only': True}
        # a provider protocol failure is never capability evidence (pilot rule):
        # transport/HTTP failures and terminal generation errors (e.g. an empty
        # visible response) are retried once in pass 2 and reported separately.
        provider_failure = bool(record.get('infrastructure_error')) or \
            record.get('finish_reason') == 'generation_error'
        record.update(sample_idx=idx, attempt=attempt, api_calls=len(calls),
                      provider_protocol_failure=provider_failure,
                      empty_visible_response=any(c['empty_visible_action'] for c in calls))
        with lock:
            spent += len(calls)
            with paths[arm][0].open('a') as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'arm': arm, 'db': row['db_id'], 'k': idx, 'attempt': attempt,
                              'correct': record['correct'], 'finish': record['finish_reason'],
                              'provider_failure': provider_failure, 'calls': len(calls),
                              'spent': spent}), flush=True)
        return record

    pending = [(arm, row, idx) for arm in ARMS for row in rows for idx in range(samples)
               if (arm, row['id'], idx) not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda a: run_one(a[0], a[1], a[2], 1), pending):
            pass
    # second pass: protocol failures only, never selected on correctness;
    # the later record replaces the failed one under last-wins dedup.
    failed = []
    for arm in ARMS:
        latest = {}
        for line in paths[arm][0].read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r['arm'], r['question_id'], r['sample_idx'])] = r
        failed += [(arm, r) for key, r in sorted(latest.items()) if is_failure(r)]
    by_id = {r['id']: r for r in rows}
    if failed:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(lambda a: run_one(a[0], by_id[a[1]['question_id']], a[1]['sample_idx'], 2), failed):
                pass

    summary = {}
    for arm in ARMS:
        latest = {}
        for line in paths[arm][0].read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r['question_id'], r['sample_idx'])] = r
        records = list(latest.values())
        summary[arm] = {'episodes': len(records), 'correct': sum(r['correct'] for r in records),
                        'submitted': sum(r['submitted'] for r in records),
                        'infrastructure_error': sum(r['infrastructure_error'] for r in records),
                        'api_calls': sum(r['api_calls'] for r in records),
                        'retried_after_protocol_failure': sum(r.get('attempt') == 2 for r in records),
                        'finish_reasons': dict(Counter(r['finish_reason'] for r in records))}
    write(OUT / 'run_summary.json', {'arms': summary, 'samples_per_question': samples,
                                     'questions': len(rows), 'calls_spent': spent, 'call_budget': budget,
                                     'training_eligible': False,
                                     'purpose': 'k-sample decoder control: masked vs strict, same ten pilot questions'})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='', help='comma-separated question ids')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--budget', type=int, default=CALL_BUDGET)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    main([s for s in a.only.split(',') if s], a.samples, a.budget, a.workers)
