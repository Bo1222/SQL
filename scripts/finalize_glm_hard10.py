"""Finalize the GLM hard-10 probe: merge the main run with protocol-failure retries.

Selection rule (outcome-blind with respect to correctness):
  For each question keep the main attempt unless it ended in a provider protocol
  failure -- a transport exception or an empty visible action (finish=stop with
  zero visible content). Only such questions were retried; no question was
  retried because its answer was wrong. If every attempt is a protocol failure,
  the main attempt is kept and flagged unavailable_protocol.

Writes compatible/records_final.jsonl and merge_summary.json (provenance of the
merge only; the headline summary is produced by summarize_glm_hard10.py); no
training export.
"""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/m6/glm_hard10_20261123'
COMPAT=OUT/'compatible'
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def protocol_failure(r):
    empty=any(not (c.get('content') or '').strip() for c in r['api_calls'])
    return bool(r.get('infrastructure_error')) or empty

def main():
    main_rows=read(COMPAT/'records.jsonl')
    retry_path=COMPAT/'records_retry1.jsonl'
    retry_rows=read(retry_path) if retry_path.exists() else []
    by_main={r['id']:r for r in main_rows}
    by_retry={r['id']:r for r in retry_rows}
    assert len(by_main)==10,f"expected 10 main records, got {len(by_main)}"
    assert not set(by_retry)-set(by_main),'retry id not in main run'
    final=[]
    for qid,r in by_main.items():
        attempts=[('main',r)]+([('retry1',by_retry[qid])] if qid in by_retry else [])
        chosen_tag,chosen=attempts[0]
        for tag,cand in attempts:
            if not protocol_failure(cand):chosen_tag,chosen=tag,cand;break
        row={**chosen,'selected_from':chosen_tag,'provider_empty_response':any(not (c.get('content') or '').strip() for c in chosen['api_calls']),
             'attempts':[{'tag':tag,'finish_reason':a['finish_reason'],'correct':a['correct'],
                          'protocol_failure':protocol_failure(a),'infrastructure_error':a.get('infrastructure_error',False),
                          'api_calls':len(a['api_calls'])} for tag,a in attempts],
             'selection_rule':'first non-protocol-failure attempt; retries were triggered by protocol failure only, not by wrong answers',
             'unavailable_protocol':all(protocol_failure(a) for _,a in attempts),
             'training_eligible':False,'diagnostic_only':True}
        final.append(row)
    with (COMPAT/'records_final.jsonl').open('w') as fh:
        for r in final:fh.write(json.dumps(r,ensure_ascii=False)+'\n')
    clean=[r for r in final if r['audit_status'] in ('supported','supported_with_scope_note')]
    summary={'n':len(final),'correct_gold_set':sum(r['correct'] for r in final),
        'submitted':sum(r['finish_reason']=='answer_submitted' for r in final),
        'unavailable_protocol':sum(r['unavailable_protocol'] for r in final),
        'retried':sorted(r['id'] for r in final if len(r['attempts'])>1),
        'selected_from_retry':sorted(r['id'] for r in final if r['selected_from']!='main'),
        'clean_subset_n':len(clean),'clean_subset_correct':sum(r['correct'] for r in clean),
        'finish_reasons':dict(Counter(r['finish_reason'] for r in final)),
        'by_audit_status':{s:{'n':sum(r['audit_status']==s for r in final),'correct':sum(r['correct'] and r['audit_status']==s for r in final)}
                           for s in sorted({r['audit_status'] for r in final})},
        'per_question':[{'db_id':r['db_id'],'id':r['id'],'audit_status':r['audit_status'],'selected_from':r['selected_from'],
                         'finish_reason':r['finish_reason'],'correct':r['correct'],'attempts':r['attempts']} for r in final],
        'training_eligible':False,'no_sft_export':True,
        'note':'Protocol failures are provider-side (transport error or empty visible action) and carry no information about SQL capability; they are not counted as teacher failures.'}
    (OUT/'merge_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('correct_gold_set','clean_subset_correct','clean_subset_n','unavailable_protocol','selected_from_retry')},ensure_ascii=False))

if __name__=='__main__':main()
