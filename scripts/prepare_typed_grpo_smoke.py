"""Prepare two RL questions for an engineering check, never an EX claim.

Choose fully graded n8 mixed groups with 2..6 successes, no reviewed input
conflict, and short traces. Outcome-based selection is intentional for this
gradient/synchronization check and cannot support benchmark generalization.
Generated student SQL is not copied into either prompts or rewards.
"""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime_contract import CONTRACT_ID, FINGERPRINT


def main():
    cal = ROOT / 'reports/m6/rl_signal_20261118'
    target = ROOT / 'reports/m6/rl_preflight_20261118'
    summary = json.loads((cal/'summary.json').read_text())
    assert summary['complete'] and summary['n_rollouts']==800
    groups = [json.loads(s) for s in (cal/'question_groups.jsonl').read_text().splitlines()]
    records = [json.loads(s) for s in (cal/'records.jsonl').read_text().splitlines()]
    excluded = {json.loads(s)['id'] for s in (target/'annotation_quarantine.jsonl').read_text().splitlines()}
    lengths = {g['id']:max(r['response_tokens']+r['observation_tokens'] for r in records if r['id']==g['id']) for g in groups}
    candidates = [g for g in groups if g['category']=='mixed' and 2<=g['successes']<=6
                  and g['id'] not in excluded and lengths[g['id']]<=1200]
    candidates.sort(key=lambda g:(abs(g['successes']-4), lengths[g['id']], g['id']))
    chosen = []
    for g in candidates:
        if g['db_id'] not in {c['db_id'] for c in chosen}: chosen.append(g)
        if len(chosen)==2: break
    if len(chosen)!=2: raise RuntimeError('not enough eligible gradient-check questions')
    parent = ROOT/'artifacts/datasets/rl_A_typed_v1_reviewed.parquet'
    frame = pd.read_parquet(parent)
    qids = {g['id'] for g in chosen}
    frame = frame[frame['extra_info'].map(lambda info:info['question_id'] in qids)]
    assert len(frame)==2
    output=target/'smoke_pool.parquet'
    if output.exists(): raise ValueError('smoke pool already prepared')
    frame.to_parquet(output,index=False)
    manifest = {'purpose':'two-update engineering check, outcome-selected training questions; not benchmark evidence',
        'runtime_contract':CONTRACT_ID,'runtime_fingerprint':FINGERPRINT,
        'source':str(parent),'source_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'questions':[{**g,'observed_max_response_tokens':lengths[g['id']]} for g in chosen],
        'selection_rule':__doc__,'generated_sql_used_for_training':False,
        'updates':2,'prompts_per_update':1,'rollouts_per_prompt':4,'lr':1e-6,'kl_loss_coef':0.001,
        'save_checkpoint':False,'wandb':'disabled'}
    (target/'smoke_pool.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__': main()
