"""Collect masked teacher traces: the ten-question pilot and the 2c expansion.

Same prompt, catalog, tools, observation encoder, budgets and student tokenizer
as the student episode; the provider's prose is removed by token mask after the
fact (`agent/teacher_trace.py`), never suppressed by the prompt. Writes
traces.jsonl (provenance: raw text, mask, action, observation), sft_pairs.jsonl
(context/action pairs for distillation) and a manifest. No model training.

Two modes:
  * default / --limit N        -- the original Phase1A pilot (first N of the
    eleven seen tasks) into trace_masked/; same behaviour as the archived run:
    one episode per row, sequential.
  * --pool FILE --repeats k    -- the 2c stratified expansion (GOAL.md §3): rows
    are the question ids listed in the pool file, resolved against
    step0_step1/label_audit.jsonl (which holds question/evidence/gold/db_path),
    k episodes each, into --out. At load time every pool row is checked against
    the pre-registration: training-eligible under the D2 disposition, not a
    seen-11 question, not a held-out-20 database.

Resume-safe by (question_id, sample_idx): a triple whose latest record is a
retryable protocol failure (generation_error / no_action_limit /
infrastructure_error) is retried once in a second pass; nothing is ever selected
on correctness. The pre-registered call budget stops new episodes and reports
`budget_exhausted` instead of overshooting. summary.json / EXPANSION.md for a
pool run are written by scripts/summarize_teacher_expansion.py.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.runtime_contract import CONTRACT_ID,FINGERPRINT,config,load_store,prompt
from agent.teacher_trace import run_teacher_episode,sft_pairs,trace_record
from eval.harness_audit import AuditTokenizer
from reward.typed_rl_reward import compute_score_with_details
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

OUT=ROOT/'reports/m6/glm_hard10_20261123/trace_masked'
SOURCE=ROOT/'reports/m6/sft_first_20261120/self_sampling/tasks_11_seen.jsonl'
AUDIT=ROOT/'reports/m6/glm_hard10_20261123/audit.jsonl'
LABEL_AUDIT=ROOT/'reports/m6/sft_first_20261120/step0_step1/label_audit.jsonl'
DISPOSITION=ROOT/'reports/m6/teacher_pool_20261123/label_disposition/label_disposition.jsonl'
SPLIT=ROOT/'reports/m6/sft_first_20261120/self_sampling/split_manifest.json'
ENDPOINT='https://tecoapi.thuwayteczz.com/v1/chat/completions'
MODEL='WAIC-GLM-5.3-Flash'
TOKENIZER=ROOT/'outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model'
MAX_TOKENS=4096
# No `stop` parameter: the provider must return its response in full, closing
# tags included. Nothing is constrained at generation time; the first complete
# action block is taken afterwards by the token mask. (An earlier run sent
# stop=[closing tags] and lost complete answers to the standard stop semantics:
# the returned content never contains the stop string, so the closing tag was
# eaten server-side and prose-prefixed answers were misread as having no action.)
STOP=None
# Protocol failures that justify exactly one re-run (never selected on correctness).
RETRYABLE={'generation_error','no_action_limit','infrastructure_error'}
DIFFERENCES=['No prose suppression by prompt; prose is removed by token mask after generation.',
    'A response without a complete action is re-sent (<=3 times) instead of consuming a round and an invalid-action slot.',
    'Several actions in one response keep the first and drop the rest; dropped actions are never executed.',
    'Provider reasoning budget is not compute-matched to the student; visible actions stay under the 512-token student budget.']

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def tasks():
    rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    return [{k:r[k] for k in ('id','db_id','db_path','question','evidence','gold_sql','metadata_path','split')}
            for r in rows if r['id']!='bird_train_e2562b2e5c5a16d031be']

def pool_rows(pool_path,only):
    """Resolve a pool file of ids into full rows, enforcing the 2c pre-registration."""
    frame=read_jsonl(pool_path)
    audit={r['question_id']:r for r in read_jsonl(LABEL_AUDIT)}
    disposition={r['question_id']:r for r in read_jsonl(DISPOSITION)}
    split=json.loads(SPLIT.read_text())
    seen={t['id'] for t in split['seen']}; heldout={t['db_id'] for t in split['heldout']}
    rows=[]
    for f in frame:
        qid=f['question_id']
        if only and qid not in only:continue
        if qid not in audit:raise SystemExit('pool id absent from label_audit.jsonl: '+qid)
        a=audit[qid];status=disposition.get(qid,{}).get('disposition') or f.get('disposition')
        if not str(status).startswith('train_eligible'):
            raise SystemExit('pool id not training-eligible under D2: '+qid+' -> '+str(status))
        if qid in seen:raise SystemExit('pool id is one of the seen-11 questions: '+qid)
        if not a.get('gold_sql'):raise SystemExit('pool id has no gold_sql: '+qid)
        if a['db_id'] in heldout:raise SystemExit('pool db is a held-out-20 database: '+a['db_id'])
        rows.append({'id':qid,'db_id':a['db_id'],'db_path':a['db_path'],'question':a['question'],
                     'evidence':a.get('evidence') or '','gold_sql':a['gold_sql'],'split':'train',
                     'audit_status':status,'semantic_subclass':a.get('semantic_subclass'),
                     'stratum':f.get('stratum'),'stratum_of_4':f.get('valid_of_4')})
    if not rows:raise SystemExit('pool selection is empty')
    return rows

def is_failure(record):
    return bool(record.get('infrastructure_error')) or record.get('finish_reason') in RETRYABLE

def collect(a):
    from dotenv import dotenv_values
    key=dotenv_values(ROOT/'.env').get('GLM_API_KEY')
    if not key:raise RuntimeError('GLM_API_KEY missing')
    only=[s for s in a.only.split(',') if s]
    pool_path=Path(a.pool) if a.pool else None
    if pool_path and not pool_path.is_absolute():pool_path=ROOT/pool_path
    out=Path(a.out) if a.out else (ROOT/'reports/m6/teacher_pool_20261123/expansion_50' if pool_path else OUT)
    if not out.is_absolute():out=ROOT/out
    if pool_path:
        rows=pool_rows(pool_path,only)
    else:
        rows=[r for r in tasks()[:a.limit] if not only or r['id'] in only]
        audit={r['id']:r['audit_status'] for r in map(json.loads,AUDIT.read_text().splitlines())}
        for r in rows:r['audit_status']=audit.get(r['id'])
    repeats=a.repeats
    tok=AuditTokenizer(TOKENIZER)
    out.mkdir(parents=True,exist_ok=True)
    traces_path=out/'traces.jsonl';pairs_path=out/'sft_pairs.jsonl';responses_path=out/'api_responses.jsonl'
    lock=threading.Lock();state={'spent':0,'budget_skipped':0}
    if responses_path.exists():state['spent']=len(responses_path.read_text().splitlines())
    latest={}
    if traces_path.exists():
        for r in read_jsonl(traces_path):latest[(r['id'],r.get('sample_idx',0))]=r
    done={k for k,r in latest.items() if not is_failure(r)}
    print(json.dumps({'rows':len(rows),'repeats':repeats,'triples':len(rows)*repeats,
                      'resumed':len(done),'calls_so_far':state['spent'],'budget':a.budget,
                      'out':str(out),'workers':a.workers}),flush=True)

    def make_generate(row,idx,attempt,calls):
        def generate_raw(ids,budget):
            body={'model':MODEL,'messages':[{'role':'user','content':tok.decode(ids)}],
                  'temperature':0.7,'top_p':1.0,'max_tokens':MAX_TOKENS}
            if STOP:body['stop']=STOP
            for tries in range(2):
                request=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),
                                               headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=180) as response:obj=json.loads(response.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429,500,502,503,504) or tries:raise RuntimeError('teacher HTTP status '+str(e.code)) from None
                    time.sleep(2)
                except (urllib.error.URLError,TimeoutError):
                    if tries:raise RuntimeError('teacher transport unavailable') from None
            choice=obj['choices'][0];message=choice.get('message',{})
            raw={'content':message.get('content') or '','finish_reason':choice.get('finish_reason'),
                 'usage':obj.get('usage'),'request_id':obj.get('id'),'model_returned':obj.get('model'),
                 'reasoning_characters':len(message.get('reasoning_content') or ''),'attempts':tries+1,
                 'visible_student_tokens':len(tok.encode(message.get('content') or '')),'budget':budget,
                 'empty_visible_action':not (message.get('content') or '').strip()}
            calls.append({'request':body,**raw})
            with lock:
                state['spent']+=1
                with responses_path.open('a') as log:
                    log.write(json.dumps({'id':row['id'],'sample_idx':idx,'attempt':attempt,**raw},ensure_ascii=False)+'\n')
            return raw
        return generate_raw

    def worker(row,idx,attempt,calls):
        store=load_store(ROOT/'artifacts/catalog',row['db_id'],row['db_path']);initial=prompt(row,store)
        assert row['gold_sql'] not in initial and 'CREATE TABLE' not in initial
        ep=run_teacher_episode(row,tok,make_generate(row,idx,attempt,calls),
            SchemaLookupTool(store,token_budget=1024,count_tokens=lambda s:len(tok.encode(s)),retrieval_version='E1.2'),
            SQLExecuteTool(row['db_path'],timeout=10,preview_rows=3,failure_policy='typed-v1'),config(),prompt=initial)
        record=trace_record(ep,row)
        scored=compute_score_with_details('bird','<answer>'+ep['final_sql']+'</answer>' if ep['final_sql'] else '',
                                          row['gold_sql'],{'db_path':row['db_path'],'score_mode':'bird_set'})
        record.update(score=scored,correct=scored['correct'],
                      reward_available=scored['reward_available'] and not ep.get('infrastructure_error',False),
                      api_calls=calls,sample_idx=idx,attempt=attempt,
                      audit_status=row.get('audit_status'),semantic_subclass=row.get('semantic_subclass'),
                      stratum=row.get('stratum'),stratum_of_4=row.get('stratum_of_4'),
                      model=MODEL,collected_at=datetime.now(timezone.utc).isoformat(),
                      runtime_contract=CONTRACT_ID,runtime_fingerprint=FINGERPRINT,
                      training_eligible=False,
                      training_note='third-party teacher output; enable only after the data-licensing decision is made by the user')
        return record,calls

    def run_one(row,idx,attempt):
        if a.budget and state['spent']>=a.budget:
            with lock:state['budget_skipped']+=1
            return None
        calls=[]
        try:
            record,calls=worker(row,idx,attempt,calls)
        except RuntimeError as exc:
            record={'id':row['id'],'db_id':row['db_id'],'question':row.get('question'),
                    'gold_sql':row.get('gold_sql'),'final_sql':None,'finish_reason':'generation_error',
                    'infrastructure_error':True,'infrastructure_reason':str(exc),'api_calls':calls,
                    'sample_idx':idx,'attempt':attempt,'correct':False,'reward_available':False,
                    'steps':[],'segments':[],'no_action_responses':0,'extra_actions_dropped':0,
                    'masked_action_tokens':0,'audit_status':row.get('audit_status'),
                    'semantic_subclass':row.get('semantic_subclass'),'stratum':row.get('stratum'),
                    'runtime_contract':CONTRACT_ID,'runtime_fingerprint':FINGERPRINT,
                    'training_eligible':False}
        with lock:
            with traces_path.open('a') as fh:fh.write(json.dumps(record,ensure_ascii=False)+'\n')
            pairs=0
            if record.get('segments'):
                with pairs_path.open('a') as fh:
                    for pair in sft_pairs({'segments':record['segments']},row,CONTRACT_ID,FINGERPRINT):
                        pair.update(teacher=MODEL,sample_idx=idx,audit_status=record.get('audit_status'),
                                    correct=record.get('correct'),training_eligible=False)
                        pairs+=1;fh.write(json.dumps(pair,ensure_ascii=False)+'\n')
            print(json.dumps({'db':row['db_id'],'id':row['id'][-12:],'k':idx,'attempt':attempt,
                              'finish':record['finish_reason'],'correct':record['correct'],
                              'no_action':record.get('no_action_responses',0),
                              'extra_dropped':record.get('extra_actions_dropped',0),
                              'pairs':pairs,'spent':state['spent']}),flush=True)
        return record

    pending=[(row,idx) for row in rows for idx in range(repeats) if (row['id'],idx) not in done]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(lambda t:run_one(t[0],t[1],1),pending):pass
    # second pass: retry the latest protocol failures only, never selected on correctness
    if traces_path.exists():
        latest={}
        for r in read_jsonl(traces_path):latest[(r['id'],r.get('sample_idx',0))]=r
    by_id={row['id']:row for row in rows}
    failed=[(by_id[key[0]],key[1]) for key,record in sorted(latest.items())
            if is_failure(record) and key[0] in by_id]
    if failed:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(lambda t:run_one(t[0],t[1],2),failed):pass

    write(out/'manifest.json',{'model':MODEL,'endpoint':ENDPOINT,'runtime_contract':CONTRACT_ID,'fingerprint':FINGERPRINT,
        'runtime_config':vars(config()),'temperature':0.7,'top_p':1.0,'api_max_tokens':MAX_TOKENS,'stop':STOP,
        'decoder':'No stop sequence and no prompt-side constraint: the provider response is taken verbatim, then the first complete action block is kept by token mask.',
        'stop_decoder_run':'reports/m6/glm_hard10_20261123/trace_masked/stop_decoder/ (superseded: stop=[closing tags] under standard stop semantics ate the closing tag of prose-prefixed answers, which were then misread as action-free)',
        'out':str(out.relative_to(ROOT)),'gold_visible':False,'full_schema_visible':False,
        'pool':str(pool_path.relative_to(ROOT)) if pool_path else None,
        'pool_sha256':sha256(pool_path) if pool_path else None,
        'repeats':repeats,'questions':len(rows),'episodes_planned':len(rows)*repeats,
        'call_budget':a.budget,'calls_spent':state['spent'],'budget_skipped':state['budget_skipped'],
        'budget_exhausted':bool(a.budget and state['spent']>=a.budget and state['budget_skipped']),
        'row_selection':('pool file rows, checked at load: training-eligible under D2, not in seen-11, no held-out-20 db'
                         if a.pool else f'Phase1A seen tasks minus one; first {a.limit} rows'+('; only '+a.only if only else '')),
        'audit_source':('label_disposition.jsonl (D2 dispositions)' if a.pool else '../audit.jsonl (frozen before any teacher output)'),
        'resume_key':'(question_id, sample_idx); retry once on latest-record protocol failure {generation_error, no_action_limit, infrastructure_error}, never on correctness',
        'differences_from_student_episode':DIFFERENCES,'training_eligible':False,
        'training_note':'third-party teacher output; the licensing decision belongs to the user'})
    print(json.dumps({'spent':state['spent'],'budget':a.budget,'budget_skipped':state['budget_skipped'],
                      'triples_recorded':len(latest)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--only',default='',help='comma-separated question ids')
    p.add_argument('--pool',default='',help='jsonl of {question_id} rows for the stratified expansion')
    p.add_argument('--repeats',type=int,default=1,help='episodes per question (k)')
    p.add_argument('--out',default='',help='output directory')
    p.add_argument('--budget',type=int,default=0,help='pre-registered API call ceiling (0 = unlimited)')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();collect(a)
