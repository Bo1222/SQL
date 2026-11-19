"""Explain an initial-lookup capacity shortfall without approving any reviews."""
from collections import defaultdict
from fractions import Fraction
from math import ceil
from pathlib import Path

from e1_alignment.common import read_jsonl, write_json, sha256_file
from e1_alignment.revision.policy import load_policy, policy_hash
from e1_alignment.revision.selection import features, learning_value


def main():
    art = Path("artifacts/m6/sft_data_revision")
    rep = Path("reports/m6/sft_data_revision")
    policy = load_policy(art / "data_policy.json")
    paths = [art / n for n in ("audited_originals.jsonl", "audited_new.jsonl", "audited_model.jsonl")]
    rows = [r for p in paths for r in read_jsonl(p)]
    reviews_path = rep / "individual_reviews.jsonl"
    reviews = read_jsonl(reviews_path)
    quarantined = {r["id"] for r in reviews if r.get("quarantine_question")}
    review_index = {r["trajectory_id"]: r for r in reviews}
    # Deliberately include unreviewed, repairable and rejected records here.
    # If this larger pool is infeasible, approving remaining reviews cannot fix it.
    pool = [r for r in rows if r["program_check"]["ok"] and r["id"] not in quarantined]
    lookups = [r for r in pool if features(r)["initial"] == "schema_lookup"]
    values = defaultdict(set)
    for r in lookups:
        values[r["id"]].add(learning_value(r))
    upper = sum(min(policy["question_cap"], len(v)) for v in values.values())
    ratio = Fraction(str(policy["initial_record_min"])) * Fraction(str(policy["initial_lookup_min"]))
    required = ceil(policy["min_records"] * ratio)
    ledger = read_jsonl(art / "acquisition_ledger.jsonl")
    model_candidates = read_jsonl(art / "model_candidates.jsonl")
    audited_model = read_jsonl(art / "audited_model.jsonl")
    gold_used = sum(r["kind"] == "gold" for r in ledger)
    model_used = sum(r["kind"] == "model" for r in ledger)
    audited_all = {r["trajectory_id"] for r in audited_model} == {r["trajectory_id"] for r in model_candidates}
    result = {
        "status": "infeasible_certified" if upper < required else "not_certified_by_lookup_bound",
        "measurement": "池数量实测；容量为按实测池与冻结策略计算的组合上界，不是能力评测。",
        "policy_sha256": policy_hash(policy), "program_pool_after_question_quarantine": len(pool),
        "initial_lookup_records": len(lookups), "initial_lookup_question_value_upper_bound": upper,
        "required_lookup_lower_bound": required,
        "proof": "N>=min_records; initial>=initial_record_min*N; lookup>=initial_lookup_min*initial. "
                 "Each question contributes at most question_cap distinct learning values. Other quota constraints "
                 "are relaxed, and non-passing/unreviewed individual reviews are included, so this is an upper bound.",
        "record_count_upper_bound_from_lookup": int(Fraction(upper, 1) / ratio),
        "gold_reserved": gold_used, "model_states_reserved": model_used,
        "acquisition_caps_consumed": gold_used >= policy["gold_attempt_cap"] and model_used >= policy["model_state_cap"],
        "all_saved_model_candidates_audited": audited_all,
        "grants_review_approval": False,
        "lookup_records": [{"trajectory_id": r["trajectory_id"], "id": r["id"],
                            "learning_value": learning_value(r),
                            "review_status": review_index.get(r["trajectory_id"], {}).get("status", "pending")}
                           for r in lookups],
        "input_hashes": {str(p): sha256_file(p) for p in paths + [reviews_path, art / "data_policy.json",
                                                                art / "acquisition_ledger.jsonl", art / "model_candidates.jsonl"]},
    }
    write_json(rep / "lookup_capacity_certificate.json", result)
    print({k: result[k] for k in ("status", "initial_lookup_records", "initial_lookup_question_value_upper_bound",
                                 "required_lookup_lower_bound", "acquisition_caps_consumed", "all_saved_model_candidates_audited")})


if __name__ == "__main__":
    main()
