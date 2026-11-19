"""E1-03: re-generate repair continuations from real E1 failure states.

Failure prefixes come from compact-observation RL rollouts on the RL Train
split.  The original SFT policy is placed at the recorded failure boundary and
sampled multiple times; only EX=1 continuations become positive trajectories.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from agent.sql_episode import encode_observation
from e1_alignment.common import (RL_TRAIN, append_jsonl, read_jsonl, stratified_sample,
                                 write_json, write_jsonl)
from e1_alignment.dataset import (apply_loss_mask, dedupe_key, exclusion_reason,
                                  finalize_record, is_positive, score_records)
from e1_alignment.engine import VLLMBatchEngine
from e1_alignment.runtime import Runtime, RuntimeContract, run_states

TRACE_DIRS = [
    "reports/m6/traces_lr_diag/lr1e6",
    "reports/m6/traces_lr_diag/lr2e6",
    "reports/m6/traces_lr_diag/lr5e6",
    "reports/m6/traces_lr_diag/lr5e6_20",
    "reports/m6/traces_lr_diag/smoke4",
]
DEFAULT_RAW = "artifacts/m6/e1_alignment/e1_repair_rollouts.raw.jsonl"
DEFAULT_PASS = "artifacts/m6/e1_alignment/e1_repair_rollouts.jsonl"
DEFAULT_BAD = "artifacts/m6/e1_alignment/e1_repair_badcases.jsonl"
DEFAULT_PREFIX = "artifacts/m6/e1_alignment/e1_failure_prefixes.jsonl"
DEFAULT_SUMMARY = "reports/m6/e1_alignment/e1_repair_rollouts.summary.json"


def classify_sql_error(payload):
    if not payload:
        return "unknown"
    message = str(payload.get("message") or "").lower()
    for needle, name in (("no such table", "no_such_table"), ("no such column", "no_such_column"),
                         ("ambiguous column", "ambiguous_column"), ("syntax error", "syntax_error"),
                         ("no such function", "no_such_function"), ("interrupt", "timeout"),
                         ("database is locked", "locked"), ("misuse", "misuse")):
        if needle in message:
            return name
    return payload.get("error_type") or "other"


def load_traces(dirs):
    traces, seen = [], set()
    for directory in dirs:
        path = Path(directory)
        if not path.is_dir():
            continue
        for file in sorted(path.glob("rollouts.worker-*.jsonl")):
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                trace = json.loads(line)
                key = (trace.get("question_id"), json.dumps(trace.get("tool_events") or [],
                                                            sort_keys=True, ensure_ascii=False, default=str))
                if key in seen:
                    continue
                seen.add(key)
                trace["_source_dir"] = path.name
                trace["_source_file"] = file.name
                traces.append(trace)
    return traces


def reconstruct_prefix(runtime, row, prefix_events):
    segments, rebuilt = [], []
    contract = runtime.contract
    for event in prefix_events:
        action, arguments = event["action"], event["arguments"]
        body = arguments if action == "sql_execute" else json.dumps(arguments, ensure_ascii=False)
        action_text = f"<{action}>{body}</{action}>"
        action_ids = runtime.tokenizer.encode(action_text, add_special_tokens=False)
        segments.append({"kind": "action", "ids": action_ids, "text": action_text,
                         "action": action, "arguments": arguments, "invalid": False,
                         "tool_event_index": len(rebuilt)})
        result = event.get("result") or {}
        obs_ids, obs_text = encode_observation(runtime.tokenizer, result,
                                               contract.max_observation_tokens,
                                               mode=contract.observation_mode)
        segments.append({"kind": "observation", "ids": obs_ids, "text": obs_text,
                         "tool_event_index": len(rebuilt)})
        rebuilt.append({"action": action, "arguments": arguments, "result": result})
    return segments, rebuilt


def candidate_states(traces, rows, runtime, score_index):
    """Return deduped failure-state candidates (before sampling)."""
    contract = runtime.contract
    candidates, seen = [], set()
    for trace in traces:
        if trace.get("invalid_actions"):
            continue
        question_id = trace.get("question_id")
        row = rows.get(question_id)
        if row is None:
            continue
        events = trace.get("tool_events") or []
        if not events:
            continue
        scored = score_index.get((question_id, (trace.get("final_sql") or "").strip()))
        if scored is None or scored.get("correct"):
            continue  # only unresolved failures
        errors = [index for index, event in enumerate(events)
                  if event.get("action") == "sql_execute"
                  and (event.get("result") or {}).get("status") == "error"]
        if errors:
            boundary = errors[0]
            boundary_kind = "sql_error"
            error_type = classify_sql_error(events[boundary].get("result"))
        else:
            boundary = len(events) - 1
            boundary_kind = "last_observation"
            error_type = "valid_sql_wrong_result"
        prefix_events = events[: boundary + 1]
        if len(prefix_events) >= contract.max_rounds:
            continue
        lookups = sum(1 for event in prefix_events if event.get("action") == "schema_lookup")
        executes = sum(1 for event in prefix_events if event.get("action") == "sql_execute")
        if lookups >= contract.max_schema_lookups and executes >= contract.max_sql_executes:
            continue
        if len(prefix_events) >= contract.max_tools:
            continue
        signature = json.dumps([[e.get("action"), e.get("arguments")] for e in prefix_events],
                               sort_keys=True, ensure_ascii=False, default=str)
        key = (question_id, boundary_kind, signature)
        if key in seen:
            continue
        seen.add(key)
        prompt = runtime.prompt_for_row(row)
        prompt_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
        segments, rebuilt = reconstruct_prefix(runtime, row, prefix_events)
        candidates.append({
            "state_id": f"{question_id}:{boundary_kind}:{boundary}",
            "question_id": question_id, "db_id": row["db_id"],
            "error_type": error_type, "boundary_kind": boundary_kind,
            "boundary_event_index": boundary,
            "source_trace": {"dir": trace.get("_source_dir"), "file": trace.get("_source_file"),
                             "finish_reason": trace.get("finish_reason"),
                             "trace_final_sql": trace.get("final_sql")},
            "prefix_rounds": len(prefix_events),
            "prefix_tool_calls": len(prefix_events),
            "prefix_schema_lookups": lookups, "prefix_sql_executes": executes,
            "prefix_actions": json.dumps([[e.get("action"), e.get("arguments")] for e in prefix_events],
                                         ensure_ascii=False, default=str),
            "segments": segments, "tool_events": rebuilt,
            "prompt": prompt, "prompt_ids": prompt_ids,
            "prompt_token_delta": len(prompt_ids) - int(trace.get("prompt_tokens") or 0),
            "row": row,
        })
    return candidates


def select_candidates(candidates, limit, seed=0):
    """Cover every error type, interleave databases inside each type, cap 2/question."""
    import random
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for candidate in candidates:
        by_type[candidate["error_type"]].append(candidate)

    queues = {}
    for error_type, items in by_type.items():
        buckets = defaultdict(list)
        for candidate in items:
            buckets[candidate["db_id"]].append(candidate)
        for db in buckets:
            rng.shuffle(buckets[db])
        queue, dbs = [], sorted(buckets)
        while any(buckets[db] for db in dbs):
            for db in dbs:
                if buckets[db]:
                    queue.append(buckets[db].pop())
        queues[error_type] = queue

    types = sorted(queues)
    positions = {error_type: 0 for error_type in types}
    picked, per_question = [], defaultdict(int)
    while len(picked) < limit:
        progressed = False
        for error_type in types:
            queue = queues[error_type]
            while positions[error_type] < len(queue):
                candidate = queue[positions[error_type]]
                positions[error_type] += 1
                if per_question[candidate["question_id"]] >= 2:
                    continue
                picked.append(candidate)
                per_question[candidate["question_id"]] += 1
                progressed = True
                break
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(RL_TRAIN))
    p.add_argument("--trace-dirs", nargs="*", default=TRACE_DIRS)
    p.add_argument("--limit-states", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--raw-out", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_PASS)
    p.add_argument("--bad-out", default=DEFAULT_BAD)
    p.add_argument("--prefix-out", default=DEFAULT_PREFIX)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--score-workers", type=int, default=8)
    p.add_argument("--states-per-batch", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--dry-run", action="store_true", help="build failure prefixes only")
    a = p.parse_args()

    fingerprint = json.loads(Path(a.fingerprint).read_text(encoding="utf-8"))["fingerprint_sha256"]
    rows = {row["id"]: row for row in read_jsonl(a.data)}
    traces = load_traces(a.trace_dirs)
    print(json.dumps({"traces": len(traces), "source_dirs": a.trace_dirs}), flush=True)

    contract = RuntimeContract(model_path=a.model, max_context=a.max_model_len)
    runtime = Runtime(contract)

    # Score every trace's final SQL once so only unresolved failures survive.
    from reward.bird_ex_reward import score_sql
    from concurrent.futures import ThreadPoolExecutor
    unique = {}
    for trace in traces:
        row = rows.get(trace.get("question_id"))
        sql = trace.get("final_sql")
        if row is None or not sql:
            continue
        key = (trace["question_id"], sql.strip())
        unique.setdefault(key, (row["db_path"], sql, row["gold_sql"]))

    def run(item):
        key, (db_path, sql, gold) = item
        try:
            return key, score_sql(db_path, sql, gold, timeout=30)
        except Exception as exc:
            return key, {"correct": False, "error": str(exc), "infrastructure_error": True}

    with ThreadPoolExecutor(max_workers=a.score_workers) as pool:
        score_index = dict(pool.map(run, unique.items()))
    print(json.dumps({"traces_scored": len(score_index),
                      "unresolved": sum(1 for v in score_index.values() if not v.get("correct"))}),
          flush=True)

    candidates = candidate_states(traces, rows, runtime, score_index)
    print(json.dumps({"candidate_states": len(candidates),
                      "by_error": {k: sum(1 for c in candidates if c["error_type"] == k)
                                   for k in sorted({c["error_type"] for c in candidates})}}),
          flush=True)
    selected = select_candidates(candidates, a.limit_states, a.seed)
    for candidate in selected:
        candidate.pop("row", None)
    write_jsonl(a.prefix_out, selected)
    print(json.dumps({"selected_states": len(selected),
                      "by_error": {k: sum(1 for c in selected if c["error_type"] == k)
                                   for k in sorted({c["error_type"] for c in selected})}}),
          flush=True)
    if a.dry_run:
        write_json(a.summary, {"stage": "E1-03", "dry_run": True, "traces": len(traces),
                               "candidate_states": len(candidates), "selected_states": len(selected),
                               "prefix_out": a.prefix_out})
        return

    engine = VLLMBatchEngine(a.model, max_model_len=a.max_model_len,
                             gpu_memory_utilization=a.gpu_memory_utilization)
    for path in (a.raw_out, a.out, a.bad_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
    started = time.monotonic()
    totals = {"states": 0, "candidates": 0, "positive": 0, "bad_cases": 0}
    batch_size = max(1, a.states_per_batch)
    state_counter = 0
    for chunk_start in range(0, len(selected), batch_size):
        if a.max_seconds and time.monotonic() - started > a.max_seconds:
            print("[stop] max-seconds reached", flush=True)
            break
        chunk = selected[chunk_start:chunk_start + batch_size]
        pairs = []
        for candidate in chunk:
            state_counter += 1
            row = rows[candidate["question_id"]]
            for candidate_index in range(a.n):
                state = runtime.new_state(row, prefix_segments=candidate["segments"],
                                          tool_events=candidate["tool_events"])
                state.index = len(pairs)
                state.candidate_index = candidate_index
                pairs.append((state, candidate))
        run_states([state for state, _ in pairs], engine, temperature=a.temperature,
                   top_p=a.top_p, seed_base=a.seed * 1_000_000 + state_counter * 1009)
        records = []
        for state, candidate in pairs:
            record = state.record()
            record["candidate_index"] = getattr(state, "candidate_index", 0)
            record["trajectory_source"] = "e1_repair_continuation"
            record["failure_state"] = {key: candidate[key] for key in
                                       ("state_id", "error_type", "boundary_kind", "boundary_event_index",
                                        "prefix_rounds", "prefix_tool_calls", "prefix_schema_lookups",
                                        "prefix_sql_executes", "source_trace", "prompt_token_delta")}
            record["prefix_action_segments"] = sum(1 for segment in candidate["segments"]
                                                   if segment["kind"] == "action")
            record["collection"] = {"stage": "E1-03", "temperature": a.temperature, "top_p": a.top_p,
                                    "n": a.n, "seed_base": a.seed, "state_index": state_counter,
                                    "source_policy_note": "failure state from compact RL Train rollouts"}
            records.append(record)
        score_records(records, workers=a.score_workers)
        for record in records:
            apply_loss_mask(record, prefix_action_segments=record["prefix_action_segments"])
            finalize_record(record, fingerprint, "e1_repair_continuation",
                            policy_checkpoint=str(Path(a.model).resolve()),
                            gold_used_for_construction=False,
                            extra={"prefix_action_segments": record["prefix_action_segments"]})
            record["excluded_reason"] = exclusion_reason(record)
            record["dedupe_key"] = dedupe_key(record)
            append_jsonl(a.raw_out, record)
            if is_positive(record):
                append_jsonl(a.out, record)
                totals["positive"] += 1
            else:
                append_jsonl(a.bad_out, record)
                totals["bad_cases"] += 1
        totals["states"] += len(chunk)
        totals["candidates"] += len(records)
        print(json.dumps({"states": totals["states"], "candidates": totals["candidates"],
                          "positive": totals["positive"], "bad_cases": totals["bad_cases"],
                          "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False),
              flush=True)
        write_json(a.summary, {"stage": "E1-03", "selected_states": len(selected), "n": a.n,
                               "temperature": a.temperature, "fingerprint_sha256": fingerprint,
                               **totals, "elapsed_s": round(time.monotonic() - started, 1),
                               "raw_out": a.raw_out, "out": a.out, "bad_out": a.bad_out,
                               "prefix_out": a.prefix_out})
    write_json(a.summary, {"stage": "E1-03", "selected_states": len(selected), "n": a.n,
                           "temperature": a.temperature, "fingerprint_sha256": fingerprint,
                           **totals, "elapsed_s": round(time.monotonic() - started, 1),
                           "raw_out": a.raw_out, "out": a.out, "bad_out": a.bad_out,
                           "prefix_out": a.prefix_out, "complete": True})
    print(json.dumps({"done": True, **totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
