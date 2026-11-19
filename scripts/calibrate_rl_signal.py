"""Fixed-A, frozen-step788 RL-train n8 calibration. Not training data.

100 proportionally stratified RL questions, 8 stochastic rollouts each. Gold
only grades, never selects a model action. pass@1 is the sampling estimator
sum(successes)/800, not an independently measured greedy accuracy.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.runtime_contract import CONTRACT, CONTRACT_ID, FINGERPRINT, config, load_store, prompt
from agent.sql_episode import episode_to_record, run_episode
from eval.harness_audit import AuditTokenizer
from reward.bird_ex_reward import compute_score_with_details
from scripts.eval_3b_dev200_abc import concurrent_episodes, read_jsonl, sha_file, sha_text, write_json
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

DEFAULT_OUT = "reports/m6/rl_signal_20261118"
MODEL = "outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
SOURCE = "data/processed/bird_agent/rl.jsonl"
SEED = 42
SAMPLES = 8


def stratified_candidates(rows, n=100, seed=SEED):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["db_id"]].append(row)
    n = min(n, len(rows))
    exact = {db: n * len(bucket) / len(rows) for db, bucket in buckets.items()}
    quota = {db: math.floor(count) for db, count in exact.items()}
    for db in sorted(buckets, key=lambda db: (-(exact[db] - quota[db]), db))[:n - sum(quota.values())]:
        quota[db] += 1
    rng = random.Random(seed)
    for db in sorted(buckets):
        buckets[db] = sorted(buckets[db], key=lambda row: row["id"])
        rng.shuffle(buckets[db])
    return buckets, quota


def annotation_flags(row):
    """Review candidates only; no automatic semantic label or data exclusion."""
    evidence, gold = row.get("evidence", ""), row["gold_sql"]
    pattern = r'([A-Za-z_]\w*)\s*(<=|>=|≤|≥|<|>|=)\s*(-?\d+(?:\.\d+)?)'
    evidence_comparisons = re.findall(pattern, evidence)
    gold_comparisons = re.findall(pattern, gold)
    flags = []
    for col, op, value in evidence_comparisons:
        op = {"≤": "<=", "≥": ">="}.get(op, op)
        for gcol, gop, gvalue in gold_comparisons:
            if col.lower() == gcol.lower() and value == gvalue and (op, gop) in {
                    ("<=", "<"), (">=", ">"), ("<", "<="), (">", ">=")}:
                flags.append({"kind": "boundary_operator_disagreement", "column": col,
                              "evidence_operator": op, "gold_operator": gop, "value": value})
    for col, value in re.findall(r"([A-Za-z_]\w*)\s*=\s*'([^']+)'", evidence):
        for gcol, gvalue in re.findall(r"([A-Za-z_]\w*)\s*=\s*'([^']+)'", gold):
            if col.lower() == gcol.lower() and value.lower() == gvalue.lower() and value != gvalue:
                flags.append({"kind": "literal_case_disagreement", "column": col,
                              "evidence_value": value, "gold_value": gvalue})
    return flags


def prepare(out):
    if out.exists():
        raise ValueError("output exists; choose a fresh --out or run to resume")
    rows = read_jsonl(ROOT / SOURCE)
    assert len(rows) == 1801 and len({r["id"] for r in rows}) == len(rows)
    assert all(r["split"] == "rl" for r in rows)
    evals = ["artifacts/datasets/dev200.jsonl", "data/processed/bird_agent/val.jsonl",
             "data/processed/bird_agent/test.jsonl"]
    for path in evals:
        holdout = read_jsonl(ROOT / path)
        assert not ({r["id"] for r in rows} & {r["id"] for r in holdout})
        assert not ({r["db_id"] for r in rows} & {r["db_id"] for r in holdout})
    sft_ids = {r["id"] for r in read_jsonl(ROOT / "data/processed/bird_agent/sft.jsonl")}
    assert not ({r["id"] for r in rows} & sft_ids)
    buckets, quota = stratified_candidates(rows)
    selected, screening = [], []
    # Replace only unusable Golds within the same DB quota, never based on
    # student outcomes. The complete screening receipt stays in the report.
    for db in sorted(buckets):
        needed = quota[db]
        for row in buckets[db]:
            if not needed:
                break
            scored = compute_score_with_details("bird", "<answer>" + row["gold_sql"] + "</answer>",
                                                row["gold_sql"], {"db_path": row["db_path"], "score_mode": "bird_set"})
            usable = scored["reward_available"] and scored["correct"]
            screening.append({"id": row["id"], "db_id": db, "gold_usable": usable, "score": scored})
            if usable:
                selected.append(row);needed -= 1
        if needed:
            raise ValueError(f"not enough scoreable Golds for {db} quota")
    assert len(selected) == 100
    rng = random.Random(SEED)
    rng.shuffle(selected)
    tokenizer = AuditTokenizer(ROOT / MODEL)
    stores, prompts = {}, []
    for index, row in enumerate(selected):
        db = row["db_id"]
        if db not in stores:
            stores[db] = load_store(ROOT / "artifacts/catalog", db, row["db_path"])
        text = prompt(row, stores[db]); ids = tokenizer.encode(text)
        if len(ids) > 2048:
            raise ValueError("prompt exceeds the shared RL prompt tensor budget")
        prompts.append({"id": row["id"], "db_id": db, "prompt": text,
                        "prompt_sha256": sha_text(text), "prompt_tokens": len(ids), "task_index": index})
    code_paths = [ROOT / p for p in ["scripts/calibrate_rl_signal.py", "scripts/eval_3b_dev200_abc.py",
        "agent/protocol.py", "agent/sql_episode.py", "agent/runtime_contract.py", "eval/input_ablation.py",
        "eval/harness_audit.py", "reward/bird_ex_reward.py", "tools/schema_lookup.py",
        "tools/sql_execute.py", "tools/sql_worker.py"]]
    code_paths += sorted((ROOT / "retrieval").glob("*.py"))
    code_paths += [ROOT / SOURCE] + [ROOT / p for p in evals]
    for db in stores:
        code_paths += sorted((ROOT / "artifacts/catalog" / db).glob("*.json"))
    model_files = sorted(p for p in (ROOT / MODEL).iterdir() if p.is_file())
    code_paths += model_files
    checks = {str(p.resolve()): sha_file(p) for p in code_paths}
    out.mkdir(parents=True)
    for name, data in [("tasks.jsonl", selected), ("prompts.jsonl", prompts), ("gold_screening.jsonl", screening)]:
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
    all_flags = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"],
                  "evidence": r.get("evidence", ""), "gold_sql": r["gold_sql"],
                  "flags": annotation_flags(r), "selected_for_calibration": r["id"] in {x['id'] for x in selected},
                  "status": "needs_semantic_review_not_automatically_excluded"}
                 for r in rows if annotation_flags(r)]
    (out / "annotation_flags.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_flags))
    manifest = {"diagnostic_only": True, "training_eligible": False, "model": str((ROOT / MODEL).resolve()),
                "source": SOURCE, "source_rows": 1801, "n_questions": 100, "n_samples": 8, "n_episodes": 800,
                "seed": SEED, "runtime_contract": CONTRACT_ID, "runtime_config": CONTRACT,
                "runtime_fingerprint": FINGERPRINT, "selection": "proportional DB quotas via largest remainders; per-DB seeded shuffle; replace only unscoreable Golds",
                "db_quota": quota, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "seed_formula": "42 + task_index*1000 + sample_index*10 + turn_index", "max_action_tokens": 512},
                "score_mode": "bird_set", "model_inventory": {str(p.resolve()): {"size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns} for p in model_files}, "input_sha256": checks,
                "prepared_files_sha256": {name: sha_file(out / name) for name in ("tasks.jsonl", "prompts.jsonl", "gold_screening.jsonl")},
                "annotation_review_candidates": len(all_flags), "gold_screening_attempts": len(screening),
                "limits": ["RL training questions, not a benchmark or independent test.",
                "pass@1 estimates stochastic sampling success probability, not greedy EX.",
                "Finite n8 coverage is not a capability ceiling.", "Flags are candidates, not established annotation-error rates."]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "n": 100, "samples": 8, "dbs": len(stores),
                      "annotation_flags": len(all_flags), "max_prompt_tokens": max(p['prompt_tokens'] for p in prompts)}), flush=True)


def validate(out):
    m = json.loads((out / "manifest.json").read_text())
    assert m["runtime_fingerprint"] == FINGERPRINT
    for path, digest in m["input_sha256"].items():
        if path in m["model_inventory"]:
            stat = Path(path).stat();old = m["model_inventory"][path]
            if stat.st_size != old["size"] or stat.st_mtime_ns != old["mtime_ns"]:
                raise ValueError("model file changed")
        elif sha_file(path) != digest:
            raise ValueError(f"input changed: {path}")
    for name, digest in m["prepared_files_sha256"].items():
        if sha_file(out / name) != digest: raise ValueError(f"prepared file changed: {name}")
    return m


def run(out):
    m = validate(out)
    tasks = read_jsonl(out / "tasks.jsonl")
    prompts = {p['id']: p for p in read_jsonl(out / "prompts.jsonl")}
    prior = read_jsonl(out / "records.jsonl")
    done = {(r['id'], r['sample_index']) for r in prior}
    assert len(done) == len(prior)
    assert done <= {(row['id'], s) for row in tasks for s in range(8)}
    assert all(r['runtime_fingerprint'] == FINGERPRINT and
               r['prompt_sha256'] == prompts[r['id']]['prompt_sha256'] for r in prior)
    jobs = [(row, i, s) for i, row in enumerate(tasks) for s in range(8) if (row['id'], s) not in done]
    if not jobs: return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(m['model'], local_files_only=True)
    started = time.monotonic()
    llm = LLM(model=m['model'], dtype="bfloat16", gpu_memory_utilization=0.55, enforce_eager=True,
              max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=8192, seed=SEED)
    stores = {row['db_id']: load_store(ROOT / "artifacts/catalog", row['db_id'], row['db_path']) for row in tasks}
    schema_tools = {db: SchemaLookupTool(store, token_budget=1024, retrieval_version="E1.2",
                    count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False))) for db, store in stores.items()}
    def generate_batch(requests):
        outputs = llm.generate(prompts=[{"prompt_token_ids": ids} for ids, _ in requests],
            sampling_params=[SamplingParams(max_tokens=opts[0], temperature=1.0, top_p=1.0,
                                            top_k=-1, seed=opts[1]) for _, opts in requests], use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]
    def worker(task, enqueue):
        row, index, sample_index = task
        turn = 0
        def generate(ids, remaining):
            nonlocal turn
            seed = SEED + index * 1000 + sample_index * 10 + turn
            turn += 1
            return enqueue(ids, (remaining, seed))
        state = run_episode(row, tokenizer, generate, schema_tools[row['db_id']],
                    SQLExecuteTool(row['db_path'], timeout=10, preview_rows=3, failure_policy="typed-v1"),
                    config(), prompt=prompts[row['id']]['prompt'])
        record = episode_to_record(state, row)
        scored = compute_score_with_details("bird", "<answer>" + state['final_sql'] + "</answer>" if state['final_sql'] else "",
                    row['gold_sql'], {"db_path": row['db_path'], "score_mode": "bird_set"})
        available = scored['reward_available'] and not record['infrastructure_error']
        record.update({"sample_index": sample_index, "task_index": index, "temperature": 1.0,
            "score": scored, "reward_available": available, "correct": bool(scored['correct'] and available),
            "valid": scored['ok'], "runtime_contract": CONTRACT_ID, "runtime_fingerprint": FINGERPRINT,
            "training_eligible": False, "diagnostic_only": True,
            "prompt_sha256": prompts[row['id']]['prompt_sha256'], "prompt_tokens": prompts[row['id']]['prompt_tokens'],
            "response_tokens": sum(len(s['ids']) for s in state['segments'] if s['kind']=='action'),
            "observation_tokens": sum(len(s['ids']) for s in state['segments'] if s['kind']=='observation')})
        return record
    completed = len(prior)
    with (out / "records.jsonl").open('a') as stream:
        for record in concurrent_episodes(jobs, worker, generate_batch):
            stream.write(json.dumps(record, ensure_ascii=False, default=str)+'\n');stream.flush()
            completed += 1
            print(json.dumps({"completed": completed, "of": 800, "id": record['id'],
                "sample": record['sample_index'], "correct": record['correct'],
                "reward_available": record['reward_available'], "elapsed_s": round(time.monotonic()-started,1)}), flush=True)
    validate(out)
    write_json(out / "runtime.json", {"wall_seconds_including_engine_load": time.monotonic()-started,
        "episodes_generated": len(jobs), "seed": SEED, "completed": True})


def summarize(out):
    m = validate(out);records = read_jsonl(out / 'records.jsonl');tasks = read_jsonl(out/'tasks.jsonl')
    groups = defaultdict(list)
    for r in records: groups[r['id']].append(r)
    assert set(groups)=={r['id'] for r in tasks} and len(records)==800
    cases=[];counts=Counter();successes=0;unique_sql=set()
    for task in tasks:
        rs=sorted(groups[task['id']], key=lambda r:r['sample_index'])
        assert [r['sample_index'] for r in rs]==list(range(8))
        available=all(r['reward_available'] for r in rs);n=sum(r['correct'] for r in rs)
        category='unavailable' if not available else ('all_zero' if n==0 else ('all_correct' if n==8 else 'mixed'))
        counts[category]+=1
        if available:successes+=n
        for r in rs:
            if r['correct']: unique_sql.add((r['id'],re.sub(r'\s+',' ',r['final_sql']).strip().rstrip(';')))
        cases.append({'id':task['id'],'db_id':task['db_id'],'category':category,'successes':n,
                      'n':8,'annotation_flags':annotation_flags(task)})
    graded=100-counts['unavailable']
    summary={'complete':True,'n_questions':100,'n_rollouts':800,'group_counts':dict(counts),
        'fully_graded_groups':graded,'sampling_pass_at_1':successes/(graded*8) if graded else None,
        'sampling_pass_at_8':(counts['mixed']+counts['all_correct'])/graded if graded else None,
        'observed_success_questions':sum(c['successes']>0 for c in cases),'unique_success_sql_texts':len(unique_sql),
        'missing_answers':sum(not r['final_sql'] for r in records),
        'unavailable_rewards':sum(not r['reward_available'] for r in records),
        'candidate_failures':sum(r['score'].get('failure_origin')=='candidate' and not r['valid'] for r in records),
        'candidate_resource_failures':sum(r['score'].get('failure_origin')=='candidate' and
             (r['score'].get('error_type')=='timeout' or r['score'].get('error') in
              ('MemoryError','interrupted','worker watchdog timeout')) for r in records),
        'runtime_contract':CONTRACT_ID,'runtime_fingerprint':FINGERPRINT,'limits':m['limits']}
    write_json(out/'summary.json',summary)
    (out/'question_groups.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','run','summarize'])
    parser.add_argument('--out',default=DEFAULT_OUT);args=parser.parse_args()
    globals()[args.phase]((ROOT/args.out).resolve())
