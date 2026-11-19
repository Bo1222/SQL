"""Frozen-seed sampling on a training diagnostic pool, before/after GRPO.

This does not select benchmark checkpoints or create SFT trajectories.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime_contract import CONTRACT_ID, FINGERPRINT, config, load_store, prompt
from agent.sql_episode import episode_to_record, run_episode
from reward.typed_rl_reward import compute_score_with_details
from scripts.eval_3b_dev200_abc import concurrent_episodes, read_jsonl, sha_file, sha_text, write_json
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--samples',type=int,default=8)
    a=p.parse_args()
    out=Path(a.out);model=Path(a.model).resolve();tasks=read_jsonl(a.tasks)
    assert len({r['id'] for r in tasks})==len(tasks) and all(r['split']=='rl' for r in tasks)
    assert a.samples>=2
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    stores={r['db_id']:load_store(ROOT/'artifacts/catalog',r['db_id'],r['db_path']) for r in tasks}
    prompts={r['id']:prompt(r,stores[r['db_id']]) for r in tasks}
    prompt_ids={qid:tokenizer.encode(s,add_special_tokens=False) for qid,s in prompts.items()}
    paths=[ROOT/q for q in ['scripts/eval_rl_learning_pool.py','scripts/eval_3b_dev200_abc.py',
        'agent/runtime_contract.py','agent/protocol.py','agent/sql_episode.py',
        'tools/sql_execute.py','tools/sql_worker.py','tools/schema_lookup.py',
        'reward/typed_rl_reward.py','reward/bird_ex_reward.py']]
    paths+=sorted((ROOT/'retrieval').glob('*.py'))
    manifest={'model':str(model),'tasks':str(Path(a.tasks).resolve()),'tasks_sha256':sha_file(a.tasks),
        'n_questions':len(tasks),'samples':a.samples,'seed':a.seed,'temperature':1.0,'top_p':1.0,'top_k':-1,
        'runtime_contract':CONTRACT_ID,'runtime_fingerprint':FINGERPRINT,'diagnostic_only':True,
        'code_sha256':{str(q):sha_file(q) for q in paths},
        'prompt_ids_sha256':{qid:sha_text(json.dumps(ids)) for qid,ids in prompt_ids.items()},
        'model_inventory':{str(q):{'size':q.stat().st_size,'mtime_ns':q.stat().st_mtime_ns}
                           for q in model.iterdir() if q.is_file()}}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==manifest,'resume inputs changed'
    else: write_json(out/'manifest.json',manifest)
    prior=read_jsonl(out/'records.jsonl');done={(r['id'],r['sample_index']) for r in prior}
    assert len(done)==len(prior)
    jobs=[(row,i,s) for i,row in enumerate(tasks) for s in range(a.samples) if (row['id'],s) not in done]
    started=time.monotonic()
    if jobs:
        from vllm import LLM, SamplingParams
        llm=LLM(model=str(model),dtype='bfloat16',gpu_memory_utilization=0.55,enforce_eager=True,
                max_model_len=8192,max_num_seqs=16,max_num_batched_tokens=8192,seed=a.seed)
        schema={db:SchemaLookupTool(store,token_budget=1024,retrieval_version='E1.2',
                    count_tokens=lambda s:len(tokenizer.encode(s,add_special_tokens=False))) for db,store in stores.items()}
        def batch_generate(requests):
            outputs=llm.generate(prompts=[{'prompt_token_ids':ids} for ids,_ in requests],
                sampling_params=[SamplingParams(max_tokens=opts[0],temperature=1.0,top_p=1.0,top_k=-1,seed=opts[1])
                                 for _,opts in requests],use_tqdm=False)
            return [list(o.outputs[0].token_ids) for o in outputs]
        def worker(job,enqueue):
            row,index,sample=job;turn=0
            def generate(ids,budget):
                nonlocal turn
                seed=a.seed+index*1000+sample*10+turn;turn+=1
                return enqueue(ids,(budget,seed))
            state=run_episode(row,tokenizer,generate,schema[row['db_id']],
                SQLExecuteTool(row['db_path'],timeout=10,preview_rows=3,failure_policy='typed-v1'),
                config(),prompt=prompts[row['id']])
            record=episode_to_record(state,row)
            scored=compute_score_with_details('bird',('<answer>'+state['final_sql']+'</answer>') if state['final_sql'] else '',
                                              row['gold_sql'],{'db_path':row['db_path'],'score_mode':'bird_set'})
            available=scored['reward_available'] and not record['infrastructure_error']
            record.update(sample_index=sample,task_index=index,score=scored,reward_available=available,
                correct=bool(available and scored['correct']),valid=scored['ok'],training_eligible=False,
                diagnostic_only=True,runtime_fingerprint=FINGERPRINT)
            return record
        with (out/'records.jsonl').open('a') as fh:
            count=len(prior)
            for r in concurrent_episodes(jobs,worker,batch_generate):
                fh.write(json.dumps(r,ensure_ascii=False)+'\n');fh.flush();count+=1
                print(json.dumps({'completed':count,'of':len(tasks)*a.samples,'correct':r['correct'],
                                  'reward_available':r['reward_available']}),flush=True)
    records=read_jsonl(out/'records.jsonl');grouped=defaultdict(list)
    for r in records:grouped[r['id']].append(r)
    assert len(records)==len(tasks)*a.samples and set(grouped)==set(prompts)
    groups=[]
    for row in tasks:
        rs=grouped[row['id']]
        assert sorted(r['sample_index'] for r in rs)==list(range(a.samples))
        n=sum(r['correct'] for r in rs);available=all(r['reward_available'] for r in rs)
        category='unavailable' if not available else ('all_zero' if n==0 else ('all_correct' if n==a.samples else 'mixed'))
        groups.append({'id':row['id'],'db_id':row['db_id'],'successes':n,'n':a.samples,'category':category})
    graded=[g for g in groups if g['category']!='unavailable']
    summary={'complete':True,'n_questions':len(tasks),'n_rollouts':len(records),
        'group_counts':dict(Counter(g['category'] for g in groups)),
        'sampling_pass_at_1':sum(g['successes'] for g in graded)/(len(graded)*a.samples) if graded else None,
        f'sampling_pass_at_{a.samples}':sum(g['successes']>0 for g in graded)/len(graded) if graded else None,
        'unavailable_rewards':sum(not r['reward_available'] for r in records),
        'wall_seconds_this_invocation':time.monotonic()-started,'diagnostic_only':True}
    write_json(out/'summary.json',summary)
    (out/'question_groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
