"""M6-08k data-only CLI. Never invokes the old train/eval/RL pipeline.

Run preflight, audit, construct, audit-new, select. Reviews are supplied through
the content-bound individual-review ledger, never inferred from EX or quotas.
"""
import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from e1_alignment.common import (RL_TRAIN, read_jsonl, write_json, write_jsonl,
                                 append_jsonl, canonical_hash, sha256_file)
from e1_alignment.rebuild.quality import content_hash, versioned, failure_prefix_check
from e1_alignment.rebuild.freeze import isolation_inputs
from e1_alignment.revision.policy import load_policy, policy_hash
from e1_alignment.revision.provenance import pointer, resolve, history, events, match_prefix
from e1_alignment.revision.quality import prepare, check, verified_repair
from e1_alignment.revision.construct import score_exact, build
from e1_alignment.revision.budget import reserve, original_identity
from e1_alignment.revision.selection import distribution, feasible, select
from e1_alignment.revision.freeze import code_hashes, freeze, offline_load

ART = Path("artifacts/m6/sft_data_revision")
REP = Path("reports/m6/sft_data_revision")
POLICY = ART / "data_policy.json"
PREFLIGHT = REP / "preflight.json"
OLD = Path("artifacts/m6/e1_alignment")
INPUTS = [OLD / name for name in ("e1_self_rollouts.jsonl", "e1_repair_rollouts.jsonl", "e1_gold_assisted.jsonl",
                                  "e1_self_rollouts.raw.jsonl", "e1_repair_rollouts.raw.jsonl", "e1_failure_prefixes.jsonl")]
INPUTS += [Path("artifacts/m6/sft_rebuild") / name for name in ("new_B.jsonl", "new_B.raw.jsonl", "new_B_prefixes.jsonl", "monitor64.jsonl")]
CANDIDATES = [INPUTS[0], INPUTS[1], INPUTS[2], INPUTS[6]]


def immutable_json(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError("immutable_artifact_changed:" + str(path))
    else:
        write_json(path, data)


def preflight():
    import sqlglot
    from e1_alignment.rebuild.ast_check import PARSER_VERSION
    from e1_alignment.rebuild.contract import verify_contract
    if sqlglot.__version__ != PARSER_VERSION:
        raise ValueError("fixed_sqlglot_required")
    p = load_policy()
    immutable_json(POLICY, p)
    # Every data job requires the same runtime, DBs, tokenizer and source weights.
    fingerprint = verify_contract()
    paths = INPUTS + isolation_inputs() + [Path(p["runtime_path"])]
    inputs = {str(path.resolve()): sha256_file(path) for path in paths}
    snapshot = {"format": "sqlrl-m6-08k-preflight-v1", "python": sys.executable,
                "sqlglot_version": sqlglot.__version__, "data_policy_sha256": policy_hash(p),
                "runtime_fingerprint_sha256": fingerprint, "inputs": inputs, "code_hashes": code_hashes(),
                "operations": p["authorized_operations"], "training_authorized": False}
    immutable_json(PREFLIGHT, snapshot)
    for path in Path("e1_alignment/revision").glob("*.py"):
        target = ART / "code_snapshot" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
    return snapshot


def context():
    p = load_policy(POLICY)
    frozen = json.loads(PREFLIGHT.read_text())
    if code_hashes() != frozen["code_hashes"] or policy_hash(p) != frozen["data_policy_sha256"]:
        raise ValueError("preflight_code_or_policy_changed")
    for path, sha in frozen["inputs"].items():
        if sha256_file(path) != sha:
            raise ValueError("preflight_input_changed:" + path)
    train = {r["id"]: r for r in read_jsonl(RL_TRAIN)}
    monitor = {r["id"] for r in read_jsonl(p["monitor_path"])}
    forbidden = {r["db_id"] for path in isolation_inputs()[1:3] for r in read_jsonl(path)}
    return p, train, monitor, frozen["runtime_fingerprint_sha256"], forbidden


def seal_code():
    """Seal fixes made during tests without repeating a completed DB/hash audit.

    Legal only before the first candidate audit. Preserve the original preflight
    and code snapshot as evidence; do not refresh data or runtime attestations.
    """
    old = json.loads(PREFLIGHT.read_text())
    for path, sha in old["inputs"].items():
        if sha256_file(path) != sha:
            raise ValueError("preflight_input_changed:" + path)
    previous_sha = sha256_file(PREFLIGHT)
    # A performance-only resume keeps completed checks and all old snapshots.
    # Semantic rule changes require a new rule version and a new data branch.
    for path in ART.glob("audited_*.jsonl"):
        for row in read_jsonl(path):
            if row.get("content_sha256") != content_hash(row) or row.get("data_policy_sha256") != old["data_policy_sha256"]:
                raise ValueError("cannot_preserve_changed_audit_record")
    archive = REP / "preflight_history" / (previous_sha + ".json")
    immutable_json(archive, old)
    snapshot = {**old, "code_hashes": code_hashes(), "runtime_verification_reused_from": str(archive),
                "runtime_verification_reused_sha256": previous_sha}
    write_json(PREFLIGHT, snapshot)
    for path in Path("e1_alignment/revision").glob("*.py"):
        target = ART / "code_snapshot" / canonical_hash(snapshot["code_hashes"]) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return {"status": "code_sealed", "repeated_runtime_operations": False}


def raw_sources():
    return [(path, i, r) for path in (INPUTS[3], INPUTS[4], INPUTS[7]) for i, r in enumerate(read_jsonl(path), 1)]


def audit(new=False, model=False):
    from e1_alignment.runtime import Runtime
    from e1_alignment.dataset import apply_loss_mask
    from e1_alignment.revision.scoring import score_sql
    p, train, monitor, fp, forbidden = context()
    runtime = Runtime()
    sources = raw_sources()
    paths = [ART / "model_candidates.jsonl"] if model else ([ART / "revisions.jsonl"] if new else CANDIDATES)
    out = ART / ("audited_model.jsonl" if model else ("audited_new.jsonl" if new else "audited_originals.jsonl"))
    completed = read_jsonl(out)
    done = {r["trajectory_id"] for r in completed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    counts = Counter(total=len(completed), program_passed=sum(r["program_check"]["ok"] for r in completed))
    for row in completed:
        counts.update(row["program_check"]["reasons"])
    started = time.monotonic()
    def audit_one(unit):
        path, line, original = unit
        row = copy.deepcopy(original) if new or model else versioned(original, str(path.resolve()), line)
        if not new and not model:
            row["trajectory_id"] = "m6-08k:" + canonical_hash(row["origin"])[:24]
        if row["trajectory_id"] in done:
            return None
        if not new and not model:
            row["construction_source"] = "gold_assisted" if row.get("gold_used_for_construction") else "model_generated"
            row["start_mode"] = "failure_continuation" if row.get("prefix_action_segments", 0) else "question_start"
            n = row.get("prefix_action_segments", 0)
            # All A/B data must be backed by a real raw policy record.
            # B prefix additionally requires the earlier independent history.
            ref = match_prefix(row, sources, n if n else len(row["tool_events"]), complete=not n)
            row["history_origin"] = ref or row["origin"]
            if n:
                true_sources = [s for s in sources if s[0] == INPUTS[3]]
                independent = match_prefix(row, true_sources, n)
                if independent:
                    row["history_origin"] = independent
                else:
                    row["unverified_independent_failure_origin"] = True
        # Rebuild only derived scores/masks; source text/tokens/feedback stay exact.
        score_exact(row, score_sql)
        apply_loss_mask(row, row.get("prefix_action_segments", 0))
        repair = verified_repair(row, score_sql)
        row["has_verified_failure_repair"] = repair["verified"]
        row["repair_evidence"] = repair
        if row["construction_source"] == "gold_assisted":
            row["autonomous_repair"] = False
        row = prepare(row, p)
        row["program_check"] = check(row, p, train, monitor, fp, runtime, forbidden)
        if row.get("unverified_independent_failure_origin"):
            row["program_check"]["reasons"].append("independent_failure_origin_not_proven")
            row["program_check"]["ok"] = False
        return row

    from concurrent.futures import ThreadPoolExecutor
    units = [(path, line, row) for path in paths for line, row in enumerate(read_jsonl(path), 1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(audit_one, units):
            if row is None:
                continue
            append_jsonl(out, row)
            counts["total"] += 1
            counts["program_passed"] += int(row["program_check"]["ok"])
            counts.update(row["program_check"]["reasons"])
            if counts["total"] % 25 == 0:
                print(json.dumps(dict(counts)), flush=True)
    report = {"measurement": "实测", "counts": dict(counts), "elapsed_s": time.monotonic() - started,
              "distribution": distribution(read_jsonl(out)), "all_program_passes_pending_individual_review": True}
    write_json(REP / ("audit_model.json" if model else ("audit_new.json" if new else "audit_originals.json")), report)
    return report


def construct():
    from e1_alignment.runtime import Runtime
    from e1_alignment.revision.scoring import score_sql
    p, train, monitor, fp, forbidden = context()
    runtime = Runtime()
    originals = read_jsonl(ART / "audited_originals.jsonl")
    quarantined = {r["id"] for r in read_jsonl(REP / "individual_reviews.jsonl") if r.get("quarantine_question")}
    from e1_alignment.revision.quality import error_class
    from e1_alignment.revision.selection import features
    repair_questions = {e: {r["id"] for r in originals if r["program_check"]["ok"] and
                           r["id"] not in quarantined and features(r)["repair"] and features(r)["error"] == e}
                        for e in p["repair_errors"]}
    initial_lookup_questions = {r["id"] for r in originals if r["program_check"]["ok"] and
                                r["id"] not in quarantined and features(r)["initial"] == "schema_lookup"}
    import math
    minimum_initial_lookup = math.ceil(p["min_records"] * p["initial_record_min"] * p["initial_lookup_min"])
    # Prioritize B rejected only for coverage, then other real failed boundaries,
    # then source policy failures, and finally question-start demonstrations.
    tasks = []
    for parent in originals:
        if parent["id"] in quarantined or parent["program_check"]["ok"] or parent.get("unverified_independent_failure_origin"):
            continue
        if parent["start_mode"] == "failure_continuation":
            priority = 0 if set(parent["program_check"]["reasons"]) <= {"visible_schema_incomplete", "gold_visible_schema_incomplete"} else 1
            tasks.append((priority, parent, parent.get("prefix_action_segments", 0)))
    def find_failure(unit):
        path, line, raw = unit
        if path != INPUTS[3] or raw["id"] in monitor or raw["id"] in quarantined or raw["id"] not in train or raw["db_id"] in forbidden:
            return None
        if raw.get("infrastructure_error") or raw.get("final_infrastructure_error"):
            return None
        actions = [s for s in raw["segments"] if s.get("kind") == "action"]
        for number, action in enumerate(actions, 1):
            if action.get("action") != "sql_execute":
                continue
            failure = failure_prefix_check({**raw, "prefix_action_segments": number}, score_sql)
            if failure["ok"]:
                parent = versioned(raw, str(path.resolve()), line)
                parent["history_origin"] = pointer(path, line, raw)
                category = error_class(raw["tool_events"][action["tool_event_index"]]["result"], failure.get("score"))
                parent["planned_error_class"] = category
                priority = 2 if category in repair_questions and len(repair_questions[category]) < p["min_questions_per_error"] else 4
                return (priority, parent, number)
        return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task in pool.map(find_failure, raw_sources()):
            if task is not None:
                tasks.append(task)
    # Old C must be rerun from question, not copied observations.
    tasks += [(3, r, 0) for r in originals if r.get("trajectory_source") == "e1_gold_assisted"]
    seen = set()
    # Round robin across database/error buckets, first pass across independent
    # questions before considering another original record for the same question.
    from collections import defaultdict
    buckets = defaultdict(list)
    for item in tasks:
        priority, parent, boundary = item
        error = parent.get("planned_error_class") or (parent.get("repair_evidence") or {}).get("error_class") or (parent.get("failure_state") or {}).get("error_type", "none")
        buckets[(priority, parent["db_id"], error)].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item[1]["id"], item[1]["trajectory_id"]))
    ordered, seen_questions, later = [], set(), []
    for priority in sorted({key[0] for key in buckets}):
        keys = sorted(key for key in buckets if key[0] == priority)
        while any(buckets[key] for key in keys):
            for key in keys:
                if buckets[key]:
                    item = buckets[key].pop(0)
                    if item[1]["id"] in seen_questions:
                        later.append(item)
                    else:
                        ordered.append(item)
                        seen_questions.add(item[1]["id"])
    tasks = ordered + later
    totals = Counter()
    started = time.monotonic()
    for priority, parent, boundary in tasks:
        category = parent.get("planned_error_class")
        if priority == 2 and len(repair_questions.get(category, ())) >= p["min_questions_per_error"]:
            continue
        if priority == 3 and len(initial_lookup_questions) >= minimum_initial_lookup:
            continue
        identity = original_identity(parent)
        if identity in seen:
            continue
        seen.add(identity)
        if parent["id"] in monitor or parent["id"] in quarantined or parent["id"] not in train or parent["db_id"] in forbidden:
            continue
        # Reservation precedes all construction work and survives interruption.
        try:
            receipt = reserve(ART / "acquisition_ledger.jsonl", "gold", identity, p)
        except ValueError as exc:
            if str(exc) == "already_reserved_no_retry":
                continue
            if str(exc) == "acquisition_quota_exhausted":
                break
            raise
        tick = time.monotonic()
        source_ref = parent.get("history_origin") or parent["origin"]
        source = resolve(source_ref)
        row = train[parent["id"]]
        record, attempt = build(row, runtime, score_sql, fp, source if boundary else None, boundary)
        attempt.update(reservation=receipt, original_record=parent["origin"], history_origin=source_ref,
                       boundary=boundary, parent_version_id=parent["trajectory_id"], elapsed_s=time.monotonic() - tick)
        append_jsonl(ART / "gold_attempts.jsonl", attempt)
        totals["attempts"] += 1
        totals["constructed"] += int(record is not None)
        if record:
            if boundary and category in repair_questions:
                repair_questions[category].add(record["id"])
            if not boundary:
                initial_lookup_questions.add(record["id"])
            raw_path = ART / "acquired_gold.raw.jsonl"
            append_jsonl(raw_path, record)
            record["acquisition_receipt"] = pointer(raw_path, len(read_jsonl(raw_path)), record)
            record["history_origin"] = source_ref
            record["origin"] = parent["origin"]
            record["parent_version_id"] = parent["trajectory_id"]
            record["revision"] = 1
            record["trajectory_id"] = "m6-08k-revision:" + canonical_hash([identity, receipt])[:24]
            record["revision_boundary"] = boundary
            record["replaced_suffix_sha256"] = canonical_hash(parent["segments"][1 + 2 * boundary:])
            record["revision_diff"] = {"original": parent["origin"], "original_suffix_start_segment": 1 + 2 * boundary,
                                       "prefix_sha256": canonical_hash(history(source["segments"][:1 + 2 * boundary])) if boundary else None}
            append_jsonl(ART / "revisions.jsonl", record)
        else:
            totals[attempt["reason"]] += 1
        if totals["attempts"] % 10 == 0:
            print(json.dumps(dict(totals)), flush=True)
    report = {"measurement": "实测", "counts": dict(totals), "elapsed_s": time.monotonic() - started,
              "persistent_reservations": len(read_jsonl(ART / "acquisition_ledger.jsonl"))}
    write_json(REP / "construction.json", report)
    return report


def select_data():
    p, train, monitor, fp, forbidden = context()
    rows = read_jsonl(ART / "audited_originals.jsonl") + read_jsonl(ART / "audited_new.jsonl") + read_jsonl(ART / "audited_model.jsonl")
    program = [r for r in rows if r["program_check"]["ok"]]
    reviews_path = REP / "individual_reviews.jsonl"
    reviews = read_jsonl(reviews_path)
    quarantined = {r["id"] for r in reviews if r.get("quarantine_question")}
    picked, report = select(program, {r["trajectory_id"]: r for r in reviews}, p, quarantined)
    capacity, capacity_report = feasible([r for r in program if r["id"] not in quarantined], p)
    write_json(REP / "program_only_capacity.json", {**capacity_report, "grants_review_approval": False})
    review_index = {r["trajectory_id"]: r for r in reviews}
    write_jsonl(REP / "candidate_ledger.jsonl", [
        {"trajectory_id": r["trajectory_id"], "id": r["id"], "origin": r["origin"],
         "content_sha256": r["content_sha256"], "program_ok": r["program_check"]["ok"],
         "reasons": r["program_check"]["reasons"], "review_status": review_index.get(r["trajectory_id"], {}).get("status", "pending"),
         "question_quarantined": r["id"] in quarantined} for r in rows])
    if not picked:
        report.update(status="data_gate_failed", frozen_manifest=None, ready_for_sft_data=False,
                      program_only_capacity=capacity_report, review_count=len(reviews),
                      no_training_or_capability_evaluation=True)
        write_json(REP / "assessment.json", report)
        return report
    manifest = freeze(picked, reviews_path, POLICY, PREFLIGHT, ART / "training.jsonl", ART / "frozen_manifest.json", fp)
    loading = offline_load(ART / "training.jsonl", ART / "frozen_manifest.json")
    write_json(REP / "offline_load.json", loading)
    report.update(status="ready_for_sft_data", manifest=str(ART / "frozen_manifest.json"), no_training_or_capability_evaluation=True)
    write_json(REP / "assessment.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight", "seal-code", "audit", "construct", "audit-new", "sample", "audit-model", "select", "offline-load"])
    args = parser.parse_args()
    operations = {"preflight": preflight, "seal-code": seal_code, "audit": audit, "construct": construct,
                  "audit-new": lambda: audit(new=True), "select": select_data,
                  "sample": lambda: __import__("e1_alignment.revision.sample", fromlist=["collect"]).collect(),
                  "audit-model": lambda: audit(model=True),
                  "offline-load": lambda: offline_load(ART / "training.jsonl", ART / "frozen_manifest.json")}
    print(json.dumps(operations[args.phase](), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
