"""Optional source-model collection, reserved before requests, no total time cap."""
import contextlib
import copy
import signal
import time
from collections import defaultdict
from pathlib import Path
from e1_alignment.common import read_jsonl, append_jsonl, canonical_hash, write_json, stratified_sample
from e1_alignment.dataset import apply_loss_mask, finalize_record
from e1_alignment.revision.budget import reserve
from e1_alignment.revision.construct import score_exact
from e1_alignment.revision.provenance import pointer


@contextlib.contextmanager
def health_watchdog(seconds):
    def timeout(signum, frame):
        raise TimeoutError("infrastructure_no_progress_15_minutes")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ProgressEngine:
    def __init__(self, engine, seconds):
        self.engine, self.seconds = engine, seconds
        self.last_progress = time.monotonic()

    def generate(self, *args, **kwargs):
        remaining = self.seconds - (time.monotonic() - self.last_progress)
        if remaining <= 0:
            raise TimeoutError("infrastructure_no_progress_15_minutes")
        with health_watchdog(remaining):
            outputs = self.engine.generate(*args, **kwargs)
        if len(outputs) != len(args[0]):
            raise RuntimeError("infrastructure_generation_output_count_mismatch")
        if any(outputs):
            self.last_progress = time.monotonic()
        return outputs


def collect():
    from e1_alignment.revision.workflow import context, ART, REP, INPUTS
    from e1_alignment.revision.selection import feasible, features
    from e1_alignment.runtime import Runtime, run_states
    from e1_alignment.engine import VLLMBatchEngine
    from e1_alignment.revision.scoring import score_sql
    p, train, monitor, fp, forbidden = context()
    audited = read_jsonl(ART / "audited_originals.jsonl") + read_jsonl(ART / "audited_new.jsonl")
    # The optional stage is legal only after structural minimum feasibility fails.
    reviews = read_jsonl(REP / "individual_reviews.jsonl")
    rejected = {r["trajectory_id"] for r in reviews if r.get("status") != "pass"}
    quarantined = {r["id"] for r in reviews if r.get("quarantine_question")}
    _, capacity = feasible([r for r in audited if r["program_check"]["ok"] and
                            r["trajectory_id"] not in rejected and r["id"] not in quarantined], p)
    if capacity["status"] == "selected":
        return {"status": "not_needed_program_pool_feasible", "model_requests": 0}
    runtime = Runtime()
    initial_lookup = {r["id"] for r in audited if r["program_check"]["ok"] and
                      r["trajectory_id"] not in rejected and features(r)["initial"] == "schema_lookup"}
    rows = [r for r in train.values() if r["id"] not in monitor and r["id"] not in quarantined and r["db_id"] not in forbidden and r["id"] not in initial_lookup]
    # The measured deficit is supervised initial lookup. Prior source-policy
    # decisions can target that deficit without changing prompts or exposing
    # Gold: retry question starts where the source actually chose lookup, then
    # fill with independent questions. A prior EX or Gold never enters context.
    observed_lookup = set()
    for raw in read_jsonl(INPUTS[3]):
        actions = [s for s in raw["segments"] if s.get("kind") == "action"]
        if actions and actions[0].get("action") == "schema_lookup":
            observed_lookup.add(raw["id"])
    targeted = stratified_sample([r for r in rows if r["id"] in observed_lookup], p["model_state_cap"], p["seed"])
    targeted_ids = {r["id"] for r in targeted}
    room = p["model_state_cap"] - len(targeted)
    filler = stratified_sample([r for r in rows if r["id"] not in targeted_ids], room, p["seed"]) if room else []
    rows = targeted + filler
    write_json(REP / "model_sampling_plan.json", {"question_ids": [r["id"] for r in rows],
               "states": len(rows), "candidates_per_state": p["model_candidates_per_state"],
               "targeted_prior_source_lookup_questions": len(targeted), "capacity_evidence": capacity,
               "context_contains_gold_or_offline_correctness_hint": False,
               "selection_basis": "measured initial-lookup deficit; prior real source-policy first actions; frozen question starts"})
    started = time.monotonic()
    raw_path = ART / "source_model.raw.jsonl"
    engine = None
    count = 0
    try:
        for row in rows:
            state_key = canonical_hash([row["db_id"], row["question"], row.get("evidence", ""), "question_start"])
            try:
                receipt = reserve(ART / "acquisition_ledger.jsonl", "model", state_key, p)
            except ValueError as exc:
                if str(exc) == "already_reserved_no_retry":
                    continue
                if str(exc) == "acquisition_quota_exhausted":
                    break
                raise
            # This reservation includes startup failures, interrupted and failed requests.
            if engine is None:
                with health_watchdog(p["no_progress_seconds"]):
                    engine = ProgressEngine(VLLMBatchEngine(runtime.contract.model_path), p["no_progress_seconds"])
            states = [runtime.new_state(row) for _ in range(receipt["reserved_candidates"])]
            for index, state in enumerate(states):
                state.index = index
            tick = time.monotonic()
            run_states(states, engine, temperature=p["temperature"], top_p=p["top_p"],
                       seed_base=30000000 + receipt["ordinal"] * 1009)
            for state in states:
                raw = state.record()
                raw["collection_reservation"] = receipt
                append_jsonl(raw_path, raw)
                line = len(read_jsonl(raw_path))
                ref = pointer(raw_path, line, raw)
                record = copy.deepcopy(raw)
                score_exact(record, score_sql)
                apply_loss_mask(record)
                finalize_record(record, fp, "e1_self_rollout", runtime.contract.model_path)
                record.update(origin=ref, history_origin=ref, construction_source="model_generated", start_mode="question_start",
                              trajectory_id="m6-08k-model:" + canonical_hash(ref)[:24], revision=0, parent_version_id=None)
                append_jsonl(ART / "model_candidates.jsonl", record)
            count += 1
            write_json(REP / "model_collection.json", {"measurement": "实测", "states_this_run": count,
                       "elapsed_s": time.monotonic() - started, "last_state_elapsed_s": time.monotonic() - tick})
            if any(s.infrastructure_error for s in states):
                raise RuntimeError("source_model_infrastructure_failure")
    except Exception as exc:
        write_json(REP / "model_collection.json", {"status": "infrastructure_stopped", "error": str(exc),
                   "states_this_run": count, "elapsed_s": time.monotonic() - started,
                   "reservations_preserved": True})
        return {"status": "infrastructure_stopped", "error": str(exc)}
    return {"status": "complete", "states_this_run": count, "elapsed_s": time.monotonic() - started}
