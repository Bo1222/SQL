"""E1-02: collect autonomous E1 rollouts with the original SFT policy.

Every question contributes `--n` independent samples at the recorded collection
sampling temperature.  All candidates are written to the raw file with their
outcome; the pass file keeps complete successes only (current E1 contract).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from e1_alignment.common import (RL_TRAIN, ROOT, append_jsonl, read_jsonl, sha256_file,
                                 stratified_sample, write_json)
from e1_alignment.dataset import (apply_loss_mask, attach_scores, classify_success,
                                  dedupe_key, exclusion_reason, finalize_record,
                                  is_positive, score_records)
from e1_alignment.engine import VLLMBatchEngine
from e1_alignment.runtime import Runtime, RuntimeContract, run_states

DEFAULT_RAW = "artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl"
DEFAULT_PASS = "artifacts/m6/e1_alignment/e1_self_rollouts.jsonl"
DEFAULT_SUMMARY = "reports/m6/e1_alignment/e1_self_rollouts.summary.json"


def load_fingerprint(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["fingerprint_sha256"]


def make_states(rows, runtime, n):
    states = []
    for row in rows:
        for candidate in range(n):
            state = runtime.new_state(row)
            state.index = len(states)
            state.candidate_index = candidate
            states.append(state)
    return states


def records_for(states, args, fingerprint, batch_index):
    records = []
    for state in states:
        record = state.record()
        record["candidate_index"] = getattr(state, "candidate_index", 0)
        record["collection"] = {
            "stage": "E1-02", "temperature": args.temperature, "top_p": args.top_p,
            "n": args.n, "seed_base": args.seed, "batch_index": batch_index,
            "engine": "vllm", "model": str(Path(args.model).resolve()),
            "sampling_note": "collection sampling is not the greedy evaluation protocol",
        }
        records.append(record)
    score_records(records, workers=args.score_workers)
    for record in records:
        apply_loss_mask(record)
        finalize_record(record, fingerprint, "e1_self_rollout",
                        policy_checkpoint=str(Path(args.model).resolve()),
                        gold_used_for_construction=False)
        record["excluded_reason"] = exclusion_reason(record)
        record["dedupe_key"] = dedupe_key(record)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(RL_TRAIN))
    p.add_argument("--limit", type=int, default=0, help="number of training questions; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--questions-per-batch", type=int, default=4)
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--raw-out", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_PASS)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--score-workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=float, default=0)
    a = p.parse_args()

    fingerprint = load_fingerprint(a.fingerprint)
    rows = read_jsonl(a.data)
    sample = stratified_sample(rows, a.limit, a.seed)
    if a.resume and Path(a.raw_out).is_file():
        done = {}
        for record in read_jsonl(a.raw_out):
            done[record["id"]] = done.get(record["id"], 0) + 1
        sample = [row for row in sample if done.get(row["id"], 0) < a.n]
        print(f"[resume] remaining questions: {len(sample)}", flush=True)

    contract = RuntimeContract(model_path=a.model, max_context=a.max_model_len)
    runtime = Runtime(contract)
    engine = VLLMBatchEngine(a.model, max_model_len=a.max_model_len,
                             gpu_memory_utilization=a.gpu_memory_utilization)
    started = time.monotonic()
    batch_size = max(1, a.questions_per_batch)
    totals = {"questions": 0, "candidates": 0, "positive": 0, "scored": 0}
    if not a.resume:
        Path(a.raw_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.raw_out).write_text("", encoding="utf-8")
        Path(a.out).write_text("", encoding="utf-8")

    for start in range(0, len(sample), batch_size):
        if a.max_seconds and time.monotonic() - started > a.max_seconds:
            print("[stop] max-seconds reached", flush=True)
            break
        chunk = sample[start:start + batch_size]
        states = make_states(chunk, runtime, a.n)
        run_states(states, engine, temperature=a.temperature, top_p=a.top_p,
                   seed_base=a.seed * 1_000_000 + start * 1009)
        records = records_for(states, a, fingerprint, start // batch_size)
        for record in records:
            append_jsonl(a.raw_out, record)
            if is_positive(record):
                append_jsonl(a.out, record)
        totals["questions"] += len(chunk)
        totals["candidates"] += len(records)
        totals["positive"] += sum(1 for record in records if is_positive(record))
        elapsed = time.monotonic() - started
        rate = totals["candidates"] / max(elapsed, 1e-6)
        print(json.dumps({"questions": totals["questions"], "candidates": totals["candidates"],
                          "positive": totals["positive"], "elapsed_s": round(elapsed, 1),
                          "candidates_per_s": round(rate, 3)}, ensure_ascii=False), flush=True)
        write_json(a.summary, {"stage": "E1-02", "data": a.data, "limit": a.limit, "seed": a.seed,
                               "n": a.n, "temperature": a.temperature,
                               "fingerprint_sha256": fingerprint, **totals,
                               "elapsed_s": round(elapsed, 1), "raw_out": a.raw_out, "out": a.out})

    write_json(a.summary, {"stage": "E1-02", "data": a.data, "limit": a.limit, "seed": a.seed,
                           "n": a.n, "temperature": a.temperature,
                           "fingerprint_sha256": fingerprint, **totals,
                           "elapsed_s": round(time.monotonic() - started, 1),
                           "raw_out": a.raw_out, "out": a.out, "complete": True})
    print(json.dumps({"done": True, **totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
