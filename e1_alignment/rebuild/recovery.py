"""Same fixed real prefixes, same remaining budgets, greedy continuation per arm."""
import argparse
from pathlib import Path
import json

from e1_alignment.common import read_jsonl, write_json, write_jsonl, canonical_hash
from e1_alignment.runtime import Runtime, RuntimeContract, run_states
from e1_alignment.dataset import score_records
from e1_alignment.engine import VLLMBatchEngine
from e1_alignment.rebuild.assess import validate_complete, correct_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", required=True)
    p.add_argument("--model"); p.add_argument("--out", required=True)
    p.add_argument("--compare-control"); p.add_argument("--compare-candidate")
    a = p.parse_args()
    prefixes = read_jsonl(a.prefixes)
    ids = [r["state_id"] for r in prefixes]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit("empty_or_duplicate_failure_prefixes")
    if a.compare_control:
        control, candidate = read_jsonl(a.compare_control), read_jsonl(a.compare_candidate)
        expected = [{"state_id": r["state_id"]} for r in prefixes]
        validate_complete(control, expected, "state_id"); validate_complete(candidate, expected, "state_id")
        fixed_hash = canonical_hash(prefixes)
        if any(r.get("prefixes_sha256") != fixed_hash for r in control + candidate):
            raise SystemExit("failure_prefix_hash_mismatch")
        infra = any(r.get("final_infrastructure_error") for r in control + candidate)
        report = {"complete": True, "n": len(prefixes), "denominator": "fixed identical real failure prefixes",
                  "prefixes_sha256": fixed_hash, "control_correct": correct_count(control),
                  "candidate_correct": correct_count(candidate), "infrastructure_error": infra,
                  "pass": correct_count(candidate) >= correct_count(control) and not infra}
        write_json(a.out, report)
        return
    if not a.model or Path(a.out).exists():
        raise SystemExit("model required; no overwriting recovery arms")
    from e1_alignment.rebuild.contract import verify_contract
    from e1_alignment.rebuild.budget import active_budget
    active_budget("recovery_pair")
    fingerprint = verify_contract()
    train = {r["id"]: r for r in read_jsonl("data/processed/bird_agent/rl.jsonl")}
    runtime = Runtime(RuntimeContract(model_path=a.model))
    engine = VLLMBatchEngine(a.model)
    records = []
    for start in range(0, len(prefixes), 4):
        chunk = prefixes[start:start + 4]
        states = []
        for prefix in chunk:
            state = runtime.new_state(train[prefix["question_id"]], prefix_segments=prefix["segments"],
                                      tool_events=prefix["tool_events"])
            state.index = len(states); states.append(state)
        run_states(states, engine, temperature=0.0, top_p=1.0)
        batch = [s.record() for s in states]
        score_records(batch, workers=4)
        for record, prefix in zip(batch, chunk):
            record["state_id"] = prefix["state_id"]
            record["prefixes_sha256"] = canonical_hash(prefixes)
            record["policy_model_path"] = str(Path(a.model).resolve())
            record["runtime_fingerprint_sha256"] = fingerprint
        records += batch
        write_jsonl(a.out, records)
        print(json.dumps({"completed_prefixes": len(records), "fixed_total": len(prefixes)}), flush=True)


if __name__ == "__main__":
    main()
