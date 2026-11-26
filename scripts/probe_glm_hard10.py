"""Audit the ten Phase1A all-zero tasks, then probe an authorized GLM API.

No dataset export or model training. Gold/audit evidence is never model input.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.runtime_contract import CONTRACT_ID,FINGERPRINT,config,load_store,prompt
from agent.sql_episode import run_episode,episode_to_record
from eval.harness_audit import AuditTokenizer
from reward.typed_rl_reward import compute_score_with_details
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

OUT=ROOT/'reports/m6/glm_hard10_20261123'
SOURCE=ROOT/'reports/m6/sft_first_20261120/self_sampling/tasks_11_seen.jsonl'
ENDPOINT='https://tecoapi.thuwayteczz.com/v1/chat/completions'
MODEL='WAIC-GLM-5.3-Flash'
COMPAT=False
LIMIT=10
TOKENIZER=ROOT/'outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model'

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def tasks():
    rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    return [{k:r[k] for k in ('id','db_id','db_path','question','evidence','gold_sql','metadata_path','split')}
            for r in rows if r['id']!='bird_train_e2562b2e5c5a16d031be']

def query(row,sql):
    with sqlite3.connect(Path(row['db_path']).resolve().as_uri()+'?mode=ro',uri=True) as c:
        start=time.monotonic()
        c.set_progress_handler(lambda:int(time.monotonic()-start>30),10000)
        c.execute('PRAGMA query_only=ON')
        cur=c.execute(sql);values=cur.fetchall()
        return {'sql':sql,'columns':[v[0] for v in cur.description], 'rows':len(values),
                'preview':values[:12],'result_sha256':hashlib.sha256(json.dumps(values,default=str).encode()).hexdigest()}

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'audit.jsonl').exists(),'audit already exists'
    notes={
      'hockey':('supported','ScoringSC contains Stanley Cup scoring; G, playerID and year match the question/Evidence.'),
      'beer_factory':('supported','Gold matches the explicitly supplied signed percentage formula; sale is operationalized as PurchasePrice by Evidence.'),
      'disney':('supported','Gold removes currency formatting, casts REAL and implements the supplied gross ratio, not a percentage-times-100.'),
      'cookbook':('supported_with_scope_note','Only sea bass steak matches the ingredient search and has one recipe link; no duplicate weighting on this instance.'),
      'synthea':('supported','Patient identity join and all three measurement filters match question/Evidence.'),
      'public_review_platform':('output_layout_ambiguous','Gold pivots High/Medium/Low into one row. The question does not explicitly require pivot instead of category/count rows.'),
      'regional_sales':('evidence_date_format_conflict','Stored dates use M/D/YY; Evidence gives 1/1/2018, Gold compares 1/1/18 as TEXT. Check chronological equivalence and literal-sensitive differences.'),
      'food_inspection_2':('gold_evidence_conflict','Evidence says restaurant; Gold filters Liquor. Distinct facility types exist in the data.'),
    }
    results=[]
    for row in tasks():
        db=row['db_id'];sql=row['gold_sql'];aux=[]
        if db=='works_cycles' and 'EmailPromotion' in sql:
            status,note='gold_evidence_conflict','Evidence requires EmailPromotion=0, Gold uses 1; actual counts differ.'
            aux=[query(row,sql.replace('EmailPromotion = 1','EmailPromotion = 0')),
                 query(row,'SELECT P.EmailPromotion,COUNT(*) FROM Employee E JOIN Person P USING(BusinessEntityID) WHERE E.CurrentFlag=1 GROUP BY P.EmailPromotion')]
        elif db=='works_cycles':
            status,note='gold_question_projection_conflict','Question asks salesperson name AND territory; Gold returns territory only.'
            aux=[query(row,sql.replace('SELECT T4.Name','SELECT T3.FirstName, T3.LastName, T4.Name'))]
        else:status,note=notes[db]
        if db=='food_inspection_2':aux=[query(row,sql.replace("'Liquor'","'Restaurant'"))]
        if db=='cookbook':aux=[query(row,"SELECT I.name,COUNT(*),COUNT(DISTINCT Q.recipe_id) FROM Ingredient I JOIN Quantity Q USING(ingredient_id) WHERE lower(I.name) LIKE '%sea bass%' GROUP BY I.name")]
        if db=='regional_sales':
            normalized="date('20'||substr(OrderDate,-2)||'-'||printf('%02d',CAST(substr(OrderDate,1,instr(OrderDate,'/')-1) AS INTEGER))||'-'||printf('%02d',CAST(substr(substr(OrderDate,instr(OrderDate,'/')+1),1,instr(substr(OrderDate,instr(OrderDate,'/')+1),'/')-1) AS INTEGER)))"
            aux=[query(row,sql.replace("'1/1/18'","'1/1/2018'")),query(row,'SELECT DISTINCT OrderNumber FROM "Sales Orders" WHERE '+normalized+"> '2018-01-01'"),query(row,'SELECT OrderDate FROM "Sales Orders" LIMIT 12')]
        gold=query(row,sql)
        scored=compute_score_with_details('bird','<answer>'+sql+'</answer>',sql,{'db_path':row['db_path'],'score_mode':'bird_set'})
        result={**row,'audit_status':status,'review_note':note,'gold_execution':gold,'gold_self_score':scored,'checks':aux,
                'training_eligible':False,'original_label_modified':False}
        results.append(result)
        with (OUT/'audit.jsonl').open('a') as fh:fh.write(json.dumps(result,ensure_ascii=False)+'\n')
    assert len(results)==10
    write(OUT/'audit_summary.json',{'n':10,'statuses':dict(Counter(r['audit_status'] for r in results)),
        'all_gold_executable':all(r['gold_self_score']['score']==1 for r in results),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'note':'Frozen before teacher outcomes; no Gold corrections, no training export.'})
    write(OUT/'manifest.json',{'model':MODEL,'endpoint':ENDPOINT,'runtime_contract':CONTRACT_ID,'fingerprint':FINGERPRINT,
        'runtime_config':vars(config()),'temperature':0.7,'top_p':1.0,'n_per_question':1,'questions':10,
        'max_api_calls_without_retries':50,'max_transport_retries':1,'gold_visible':False,'full_schema_visible':False,
        'strict_parser':True,'token_budget_accounting':'Student tokenizer bounds visible action/observations/context; API max_tokens also capped to remaining action budget. Provider tokenizer/hidden reasoning may differ.',
        'training_eligible':False,'audit_sha256':hashlib.sha256((OUT/'audit.jsonl').read_bytes()).hexdigest()})
    print(json.dumps({'audit_complete':True,'statuses':dict(Counter(r['audit_status'] for r in results))}),flush=True)

def probe():
    from dotenv import dotenv_values
    key=dotenv_values(ROOT/'.env').get('GLM_API_KEY')
    if not key:raise RuntimeError('GLM_API_KEY missing')
    rows=[r for r in tasks()[:LIMIT] if not ONLY or r['id'] in ONLY]
    audit_rows={r['id']:r for r in map(json.loads,(OUT/'audit.jsonl').read_text().splitlines())}
    assert len(audit_rows)==10
    tok=AuditTokenizer(TOKENIZER)
    records_path=OUT/(f'records_{TAG}.jsonl' if TAG else 'records.jsonl')
    responses_path=OUT/(f'api_responses_{TAG}.jsonl' if TAG else 'api_responses.jsonl')
    prior=[json.loads(s) for s in records_path.read_text().splitlines()] if records_path.exists() else []
    done={r['id'] for r in prior}
    def worker(row):
        store=load_store(ROOT/'artifacts/catalog',row['db_id'],row['db_path']);initial=prompt(row,store)
        assert row['gold_sql'] not in initial and 'CREATE TABLE' not in initial
        api_calls=[]
        def generate(ids,budget):
            body={'model':MODEL,'messages':[{'role':'user','content':tok.decode(ids)}],
                  'temperature':0.7,'top_p':1.0,'max_tokens':budget,'thinking':{'type':'disabled'}}
            if COMPAT:
                body['max_tokens']=4096
                body.pop('thinking')
                body['stop']=['</schema_lookup>','</sql_execute>','</answer>']
            for attempt in range(2):
                request=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=180) as response:obj=json.loads(response.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429,500,502,503,504) or attempt:raise RuntimeError('teacher HTTP status '+str(e.code)) from None
                    time.sleep(2)
                except (urllib.error.URLError,TimeoutError):
                    if attempt:raise RuntimeError('teacher transport unavailable') from None
            choice=obj['choices'][0];message=choice.get('message',{});text=message.get('content') or ''
            raw_text=text
            restored=False
            if COMPAT and choice.get('finish_reason')=='stop':
                for tag in ('schema_lookup','sql_execute','answer'):
                    if text.lstrip().startswith('<'+tag+'>') and '</'+tag+'>' not in text:
                        text+='</'+tag+'>';restored=True;break
            api_calls.append({'request':body,'model_returned':obj.get('model'),'request_id':obj.get('id'),
                'usage':obj.get('usage'),'finish_reason':choice.get('finish_reason'),'content':text,
                'reasoning_characters':len(message.get('reasoning_content') or ''),'attempts':attempt+1,
                'visible_student_tokens':len(tok.encode(text)),'budget':budget,
                'raw_content':raw_text,'stop_delimiter_restored':restored,
                'empty_visible_action':not text.strip()})
            with responses_path.open('a') as log:
                log.write(json.dumps({'id':row['id'],**api_calls[-1]},ensure_ascii=False)+'\n')
            return tok.encode(text) # no lenient parsing or hidden truncation
        state=run_episode(row,tok,generate,SchemaLookupTool(store,token_budget=1024,count_tokens=lambda s:len(tok.encode(s)),retrieval_version='E1.2'),
                          SQLExecuteTool(row['db_path'],timeout=10,preview_rows=3,failure_policy='typed-v1'),config(),prompt=initial)
        r=episode_to_record(state,row)
        scored=compute_score_with_details('bird','<answer>'+state['final_sql']+'</answer>' if state['final_sql'] else '',row['gold_sql'],{'db_path':row['db_path'],'score_mode':'bird_set'})
        r.update(score=scored,correct=scored['correct'],reward_available=scored['reward_available'] and not r['infrastructure_error'],
                 api_calls=api_calls,audit_status=audit_rows[row['id']]['audit_status'],model=MODEL,diagnostic_only=True,training_eligible=False,
                 provider_empty_response=any(c['empty_visible_action'] for c in api_calls),attempt_tag=TAG or 'main')
        return r
    # Sequential, bounded pilot: at most ten episodes and five action calls each.
    for row in rows:
        if row['id'] in done:continue
        r=worker(row)
        with records_path.open('a') as fh:fh.write(json.dumps(r,ensure_ascii=False)+'\n')
        prior.append(r)
        print(json.dumps({'completed':len(prior),'db':row['db_id'],'correct':r['correct'],'finish':r['finish_reason'],'calls':len(r['api_calls'])}),flush=True)
        if r['api_calls'] and all(not x['content'] and x['finish_reason']=='length' for x in r['api_calls']):
            write(OUT/'provider_budget_block.json',{'reason':'Provider exhausted budget in hidden reasoning without a visible action; no evidence about SQL solvability.','completed':len(prior)})
            break
    write(OUT/(f'summary_{TAG}.json' if TAG else 'summary.json'),{'complete':len(prior)==10,'n':len(prior),'correct':sum(r['correct'] for r in prior),
        'attempt_tag':TAG or 'main','only':ONLY,
        'unavailable':sum(not r['reward_available'] for r in prior),'api_calls':sum(len(r['api_calls']) for r in prior),
        'by_audit_status':{s:{'n':sum(r['audit_status']==s for r in prior),'correct':sum(r['correct'] and r['audit_status']==s for r in prior)} for s in set(r['audit_status'] for r in prior)}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','probe']);p.add_argument('--compat',action='store_true');p.add_argument('--limit',type=int,default=10)
    p.add_argument('--only',default='',help='comma-separated question ids to re-run')
    p.add_argument('--attempt-tag',default='',help='archive this run as records_<tag>.jsonl / api_responses_<tag>.jsonl')
    a=p.parse_args()
    LIMIT=a.limit
    ONLY=[s for s in a.only.split(',') if s]
    TAG=a.attempt_tag
    if a.compat:
        COMPAT=True;parent=OUT;OUT=parent/'compatible';OUT.mkdir(exist_ok=True)
        shutil.copy2(parent/'audit.jsonl',OUT/'audit.jsonl')
        manifest=json.loads((parent/'manifest.json').read_text())
        manifest.update(provider_thinking='default enabled',api_max_tokens=4096,
            decoder='Stop at action closing delimiter; restore provider-omitted delimiter only. No extraction of extra prose or fake observations.',
            comparison_limit='Same information/tools/visible budgets, different provider reasoning budget and decoder stop rule; not compute-matched.')
        write(OUT/'manifest.json',manifest)
    globals()[a.phase]()
