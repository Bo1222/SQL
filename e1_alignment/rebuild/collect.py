"""Bounded new A/B/C acquisition; revisions consume the same sampling budget."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from e1_alignment.common import (RL_TRAIN, append_jsonl, read_jsonl, stratified_sample,
                                 write_json, write_jsonl, canonical_hash)
from e1_alignment.collect_a import make_states, records_for
from e1_alignment.collect_b import candidate_states, select_candidates
from e1_alignment.collect_c import (run_lookup, observation_segment, missing_query,
                                     make_record)
from e1_alignment.dataset import (apply_loss_mask, finalize_record, score_records,
                                  exclusion_reason, is_positive)
from e1_alignment.runtime import Runtime, RuntimeContract, run_states
from e1_alignment.rebuild.ast_check import extract, sqlite_schema
from e1_alignment.rebuild.visibility import coverage

ART = Path("artifacts/m6/sft_rebuild")
REP = Path("reports/m6/sft_rebuild")


def strict_gold_attempt(row, runtime):
    req = extract(row["gold_sql"], row["db_path"])
    if not req["ok"]:
        return None, {"reason": "gold_ast_unresolved", "ast": req}
    schema = sqlite_schema(row["db_path"])
    segments, events, lookups = [], [], []
    for number in range(runtime.contract.max_schema_lookups):
        if number == 0:
            arguments = {"query": (row["question"] + " " + row.get("evidence", "")).strip()}
        else:
            missing = cov["missing"]
            arguments = {"query": missing_query(row, missing)}
        action, result = run_lookup(runtime, row["db_id"], arguments)
        action["tool_event_index"] = len(events)
        events.append({"action": "schema_lookup", "arguments": arguments, "result": result})
        segments += [action, observation_segment(runtime, result, len(events) - 1)]
        lookups.append({"arguments": arguments, "gold_assisted_query": number > 0})
        cov = coverage(req, segments, len(segments), schema)
        if cov["ok"]:
            break
    if not cov["ok"]:
        return None, {"reason": "visible_coverage_incomplete", "ast": req, "coverage": cov,
                      "lookups": lookups, "segments": segments, "tool_events": events}
    # Use original Gold, real SQLite feedback; no AST rewrite or fabricated observation.
    result = runtime.sql_tool(row["db_path"])(row["gold_sql"])
    if result.get("status") != "ok" or result.get("failure_origin"):
        return None, {"reason": "gold_execution_failed", "result": result}
    text = f"<sql_execute>{row['gold_sql']}</sql_execute>"
    events.append({"action": "sql_execute", "arguments": row["gold_sql"], "result": result})
    segments.append({"kind": "action", "text": text,
                     "ids": runtime.tokenizer.encode(text, add_special_tokens=False),
                     "action": "sql_execute", "arguments": row["gold_sql"], "invalid": False,
                     "tool_event_index": len(events) - 1})
    segments.append(observation_segment(runtime, result, len(events) - 1))
    text = f"<answer>{row['gold_sql']}</answer>"
    segments.append({"kind": "action", "text": text,
                     "ids": runtime.tokenizer.encode(text, add_special_tokens=False),
                     "action": "answer", "arguments": row["gold_sql"], "invalid": False})
    return {"segments": segments, "tool_events": events, "lookups": lookups,
            "coverage": cov, "requirements": req}, {"reason": None}


def source_sft_prefixes(rows, runtime):
    traces = read_jsonl("artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl")
    scores = {}
    for trace in traces:
        trace["question_id"] = trace["id"]
        trace["_source_file"] = "e1_self_rollouts.raw.jsonl"
        trace["_source_dir"] = "source_sft"
        scores[(trace["id"], (trace.get("final_sql") or "").strip())] = {
            "correct": trace.get("final_correct") or trace.get("final_infrastructure_error")}
    candidates = candidate_states(traces, rows, runtime, scores)
    # Reconstructed text is only acceptable if it exactly reproduces real history.
    trace_index = {}
    for trace in traces:
        trace_index.setdefault(trace["id"], []).append(trace)
    valid = []
    for c in candidates:
        real = None
        for trace in trace_index.get(c["question_id"], []):
            if trace.get("final_sql") != c["source_trace"].get("trace_final_sql"):
                continue
            pairs = [(s.get("kind"), s.get("text")) for s in trace["segments"][1:1 + len(c["segments"])]]
            if pairs == [(s.get("kind"), s.get("text")) for s in c["segments"]]:
                real = trace; break
        if real is None:
            continue
        c["state_id"] = "source_sft:" + canonical_hash([c["question_id"], c["prefix_actions"]])[:24]
        c["segments"] = real["segments"][1:1 + len(c["segments"])]
        c["tool_events"] = real["tool_events"][:c["prefix_tool_calls"]]
        c["source_trace"]["origin_sha256"] = canonical_hash(real)
        valid.append(c)
    return select_candidates(valid, 150, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["A", "B", "C"], required=True)
    a = p.parse_args()
    from e1_alignment.rebuild.budget import active_budget
    active_budget(a.source)
    from e1_alignment.rebuild.contract import verify_contract
    verify_contract()
    ART.mkdir(parents=True, exist_ok=True); REP.mkdir(parents=True, exist_ok=True)
    raw, good = ART / f"new_{a.source}.raw.jsonl", ART / f"new_{a.source}.jsonl"
    if raw.exists() or good.exists():
        raise SystemExit("no automatic reacquisition/overwrite; counts include failed attempts")
    monitor = {r["id"] for r in read_jsonl(ART / "monitor64.jsonl")}
    if len(monitor) != 64:
        raise SystemExit("monitor must be frozen first")
    rows = [r for r in read_jsonl(RL_TRAIN) if r["id"] not in monitor]
    fingerprint = json.loads(Path("reports/m6/e1_alignment/runtime_fingerprint.json").read_text())["fingerprint_sha256"]
    runtime = Runtime(RuntimeContract())
    started = time.monotonic()
    totals = {"attempts": 0, "positive": 0, "source": a.source}
    engine = None
    if a.source == "A":
        # New training questions expand tool-decision and database coverage.
        used = {r["id"] for r in read_jsonl("artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl")}
        sample = stratified_sample([r for r in rows if r["id"] not in used], 200, 0)
    elif a.source == "B":
        sample = source_sft_prefixes({r["id"]: r for r in rows}, runtime)
        write_jsonl(ART / "new_B_prefixes.jsonl", [{k: v for k, v in r.items() if k != "row"} for r in sample])
    else:
        # C only fills unresolved A/B difficulties, never a generic alphabetic table demo.
        success_ids = {r["id"] for path in [ART / "new_A.jsonl", ART / "new_B.jsonl"] for r in read_jsonl(path)}
        failed_ids = {r["id"] for path in [ART / "new_A.raw.jsonl", ART / "new_B.raw.jsonl"]
                      for r in read_jsonl(path) if not r.get("final_correct")}
        sample = stratified_sample([r for r in rows if r["id"] in failed_ids - success_ids], 200, 0)
    if not sample:
        raise SystemExit("no valid acquisition units; no sampling budget consumed")
    if a.source in ("A", "B"):
        from e1_alignment.engine import VLLMBatchEngine
        engine = VLLMBatchEngine(runtime.contract.model_path)
    raw.touch(); good.touch()
    for start in range(0, len(sample), 4 if engine else 1):
        chunk = sample[start:start + (4 if engine else 1)]
        if a.source == "A":
            states = make_states(chunk, runtime, 4)
            run_states(states, engine, temperature=1.0, top_p=1.0, seed_base=10000000 + start * 1009)
            args = SimpleNamespace(temperature=1.0, top_p=1.0, n=4, seed=10,
                                   model=runtime.contract.model_path, score_workers=4)
            records = records_for(states, args, fingerprint, start // 4)
        elif a.source == "B":
            pairs = []
            for c in chunk:
                for i in range(4):
                    state = runtime.new_state(c["row"], prefix_segments=c["segments"], tool_events=c["tool_events"])
                    state.index = len(pairs); state.candidate_index = i
                    pairs.append((state, c))
            run_states([s for s, _ in pairs], engine, temperature=1.0, top_p=1.0, seed_base=20000000 + start * 1009)
            records = []
            for state, c in pairs:
                r = state.record()
                r["candidate_index"] = state.candidate_index
                r["failure_state"] = {k: v for k, v in c.items() if k not in ("row", "segments", "tool_events", "prompt", "prompt_ids")}
                r["prefix_action_segments"] = c["prefix_rounds"]
                records.append(r)
            score_records(records, workers=4)
            for r in records:
                apply_loss_mask(r, r["prefix_action_segments"])
                finalize_record(r, fingerprint, "e1_repair_continuation", runtime.contract.model_path)
                r["excluded_reason"] = exclusion_reason(r)
        else:
            selected, attempt = strict_gold_attempt(chunk[0], runtime)
            append_jsonl(ART / "new_C_attempts.jsonl", {"id": chunk[0]["id"], **attempt})
            records = [make_record(chunk[0], runtime, selected, fingerprint)] if selected else []
        for r in records:
            r["collection_branch"] = "M6-08j"
            append_jsonl(raw, r)
            if is_positive(r):
                append_jsonl(good, r); totals["positive"] += 1
        totals["attempts"] += len(records) if engine else 1
        totals["elapsed_s"] = round(time.monotonic() - started, 1)
        write_json(REP / f"collection_{a.source}.json", totals)
        print(json.dumps(totals), flush=True)
    totals["complete"] = True
    totals["actual_sample_units"] = len(sample)
    write_json(REP / f"collection_{a.source}.json", totals)


if __name__ == "__main__":
    main()
