"""Deterministic selection and mandatory training-time hash/review validation."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from e1_alignment.common import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl, RL_TRAIN, ROOT
from e1_alignment.rebuild import RULE_VERSION
from e1_alignment.rebuild.quality import SOURCES, content_hash, review_errors
from e1_alignment.rebuild.ast_check import AST_RULE_VERSION, PARSER_VERSION


def distribution(rows):
    source = Counter(SOURCES.get(r.get("trajectory_source"), "unknown") for r in rows)
    tokens = Counter()
    first = Counter()
    initial_unsupervised = 0
    zero_tool = 0
    by_db, by_error, by_action_tokens = Counter(), Counter(), Counter()
    for r in rows:
        s = SOURCES.get(r.get("trajectory_source"), "unknown")
        tokens[s] += sum(r.get("loss_mask", [])[1:])
        actions = [a for a in r.get("segments", []) if a.get("kind") == "action"]
        zero_tool += int(not any(a.get("action") != "answer" for a in actions))
        if s != "B" and actions:
            if any(actions[0].get("loss_mask", [])):
                first[actions[0]["action"]] += 1
            else:
                initial_unsupervised += 1
        by_db[r["db_id"]] += 1
        by_error[(r.get("failure_state") or {}).get("error_type", "none")] += 1
        for action in actions:
            by_action_tokens[action.get("action")] += sum(action.get("loss_mask", []))
    return {"n": len(rows), "by_source": dict(source), "supervised_tokens_by_source": dict(tokens),
            "zero_tool": zero_tool, "initial_supervised_actions": dict(first),
            "initial_supervised_denominator": sum(first.values()),
            "initial_unsupervised_non_b": initial_unsupervised,
            "by_db": dict(by_db), "by_error": dict(by_error),
            "supervised_tokens_by_action": dict(by_action_tokens)}


def quota_errors(rows):
    d = distribution(rows)
    n = len(rows)
    c, t, initial = d["by_source"], d["supervised_tokens_by_source"], d["initial_supervised_actions"]
    den = d["initial_supervised_denominator"]
    errors = []
    if not 400 <= n <= 800:
        errors.append("size_outside_400_800")
    if 4 * c.get("B", 0) < n:
        errors.append("b_below_25_percent")
    if 100 * c.get("C", 0) > 15 * n:
        errors.append("c_above_15_percent")
    if 5 * t.get("C", 0) > sum(t.values()):
        errors.append("c_tokens_above_20_percent")
    if 10 * d["zero_tool"] > n:
        errors.append("zero_tool_above_10_percent")
    if den == 0 or 100 * initial.get("answer", 0) > 15 * den:
        errors.append("initial_answer_above_15_percent_or_empty")
    for action in ("schema_lookup", "sql_execute"):
        if den == 0 or 4 * initial.get(action, 0) < den:
            errors.append(f"initial_{action}_below_25_percent")
    counts = Counter(r["id"] for r in rows)
    if any(v > 2 for v in counts.values()):
        errors.append("question_cap_exceeded")
    learning_keys = [(r["id"], learning_value(r)) for r in rows]
    if len(set(learning_keys)) != len(learning_keys):
        errors.append("duplicate_learning_value")
    return errors


def learning_value(row):
    # Syntactic SQL/query variants do not create a second learning value.
    actions = [s for s in row.get("segments", []) if s.get("kind") == "action"]
    failed = [s.get("offline_error_type") or "valid_sql_wrong_result" for s in actions
              if s.get("action") == "sql_execute" and not s.get("offline_ex")]
    return canonical_hash({"source": SOURCES.get(row.get("trajectory_source")),
                           "failure_type": (row.get("failure_state") or {}).get("error_type") or (failed[0] if failed else "none"),
                           "supervised_action_sequence": [s.get("action") for s in actions if any(s.get("loss_mask", []))],
                           "initial_action_supervised": bool(actions and any(actions[0].get("loss_mask", [])))})


def select(rows, reviews, seed=0):
    """MILP enforces ratios and caps, with seeded database/error round-robin ties.

    It never invents approvals, copies examples, or relaxes a minimum.
    """
    import random
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    eligible = [r for r in rows if not review_errors(r, reviews.get(r["trajectory_id"]))]
    return quota_feasibility(eligible, seed)


def quota_feasibility(eligible, seed=0):
    """Structural upper bound when called on program-only rows; grants NO review approval.

    Only `select` may supply records to freeze; freeze independently enforces reviews.
    """
    import random
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    buckets = defaultdict(list)
    for r in eligible:
        buckets[(r["db_id"], (r.get("failure_state") or {}).get("error_type", "none"))].append(r)
    rng = random.Random(seed)
    for key in sorted(buckets):
        buckets[key].sort(key=lambda r: r["trajectory_id"])
        rng.shuffle(buckets[key])
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop())
    n = len(ordered)
    if n < 400:
        return [], {"status": "insufficient_reviewed_records", "eligible": n,
                    "reasons": ["size_outside_400_800"], "eligible_distribution": distribution(ordered)}
    vectors, lower, upper = [], [], []
    def add(v, lo, hi):
        vectors.append(v); lower.append(lo); upper.append(hi)
    ones = np.ones(n)
    source = [SOURCES[r["trajectory_source"]] for r in ordered]
    add(ones, 400, 800)
    add(np.array([int(s == "B") - .25 for s in source]), 0, np.inf)
    add(np.array([int(s == "C") - .15 for s in source]), -np.inf, 0)
    add(np.array([(int(s == "C") - .2) * sum(r["loss_mask"][1:]) for s, r in zip(source, ordered)]), -np.inf, 0)
    initial = []
    zero = []
    for s, r in zip(source, ordered):
        acts = [a for a in r["segments"] if a["kind"] == "action"]
        initial.append(acts[0]["action"] if s != "B" and any(acts[0]["loss_mask"]) else None)
        zero.append(int(not any(a["action"] != "answer" for a in acts)) - .1)
    add(zero, -np.inf, 0)
    add([int(a is not None) for a in initial], 1, np.inf)
    add([int(a == "answer") - .15 * int(a is not None) for a in initial], -np.inf, 0)
    for action in ("schema_lookup", "sql_execute"):
        add([int(a == action) - .25 * int(a is not None) for a in initial], 0, np.inf)
    for q in sorted({r["id"] for r in ordered}):
        add([int(r["id"] == q) for r in ordered], 0, 2)
    groups = defaultdict(list)
    for i, r in enumerate(ordered):
        groups[(r["id"], learning_value(r))].append(i)
    for group in groups.values():
        if len(group) > 1:
            add([int(i in group) for i in range(n)], 0, 1)
    objective = -ones + np.arange(n) / (n * n * 2)
    solved = milp(objective, integrality=ones, bounds=Bounds(0, 1),
                  constraints=LinearConstraint(np.array(vectors), lower, upper),
                  options={"time_limit": 60, "mip_rel_gap": 0.0})
    if solved.x is None:
        return [], {"status": "quotas_infeasible", "eligible": n, "solver": solved.message,
                    "eligible_distribution": distribution(ordered)}
    picked = [r for r, flag in zip(ordered, solved.x) if flag > .5]
    errors = quota_errors(picked)
    return ([] if errors else picked), {"status": "rejected" if errors else "selected", "eligible": n,
                                       "reasons": errors, "distribution": distribution(picked),
                                       "seed": seed, "solver": solved.message}


def split_errors(rows, train_rows, monitor_ids, forbidden_dbs):
    errors = []
    for row in rows:
        reference = train_rows.get(row.get("id"))
        if reference is None:
            errors.append("not_in_rl_train")
            continue
        for field in ("question", "evidence", "gold_sql", "db_id", "db_path"):
            if (row.get(field) or "") != (reference.get(field) or ""):
                errors.append("train_identity_mismatch:" + field)
        if row.get("id") in monitor_ids:
            errors.append("monitor_leakage")
        if row.get("db_id") in forbidden_dbs:
            errors.append("evaluation_database_leakage")
    return sorted(set(errors))


def isolation_inputs():
    paths = [RL_TRAIN, ROOT / "data/processed/bird_agent/val.jsonl",
             ROOT / "data/processed/bird_agent/test.jsonl", ROOT / "artifacts/manifests/split_audit.json",
             ROOT / "artifacts/manifests/dev200.json"]
    if not all(p.is_file() for p in paths):
        raise ValueError("split_isolation_evidence_missing")
    return paths


def check_isolation(rows, monitor_ids):
    paths = isolation_inputs()
    train = {r["id"]: r for r in read_jsonl(paths[0])}
    forbidden = {r["db_id"] for p in paths[1:3] for r in read_jsonl(p)}
    return split_errors(rows, train, monitor_ids, forbidden)


def freeze(rows, reviews_path, dataset_path, manifest_path, monitor_path, fingerprint):
    reviews = read_jsonl(reviews_path)
    index = {r["trajectory_id"]: r for r in reviews}
    if len(index) != len(reviews):
        raise ValueError("duplicate_reviews")
    errors = quota_errors(rows)
    for row in rows:
        errors += review_errors(row, index.get(row["trajectory_id"]))
    monitor_rows = read_jsonl(monitor_path)
    if len(monitor_rows) != 64 or len({r["id"] for r in monitor_rows}) != 64:
        errors.append("monitor_must_contain_64_unique_questions")
    if {r["id"] for r in monitor_rows} & {r["id"] for r in rows}:
        errors.append("monitor_leakage")
    if len({r["trajectory_id"] for r in rows}) != len(rows):
        errors.append("duplicate_trajectory_ids")
    errors += check_isolation(rows, {r["id"] for r in monitor_rows})
    if errors:
        raise ValueError("freeze_rejected:" + ",".join(sorted(set(errors))))
    if Path(dataset_path).exists() or Path(manifest_path).exists():
        raise FileExistsError("frozen_artifacts_are_immutable")
    write_jsonl(dataset_path, rows)
    manifest = {"format": "sqlrl-m6-08j-frozen-v1", "rule_version": RULE_VERSION,
                "ast_rule_version": AST_RULE_VERSION, "parser_version": PARSER_VERSION,
                "runtime_fingerprint_sha256": fingerprint,
                "dataset_sha256": sha256_file(dataset_path), "reviews_path": str(Path(reviews_path).resolve()),
                "reviews_sha256": sha256_file(reviews_path), "monitor_path": str(Path(monitor_path).resolve()),
                "monitor_sha256": sha256_file(monitor_path),
                "record_hashes": {r["trajectory_id"]: content_hash(r) for r in rows},
                "isolation_input_hashes": {str(p): sha256_file(p) for p in isolation_inputs()},
                "distribution": distribution(rows)}
    write_json(manifest_path, manifest)
    return manifest


def verify_frozen(dataset_path, manifest_path):
    version = json.loads(Path(manifest_path).read_text()).get("format")
    if version == "sqlrl-m6-08k-frozen-v1":
        from e1_alignment.revision.freeze import verify_frozen as verify_revision
        return verify_revision(dataset_path, manifest_path)
    import sqlglot
    if sqlglot.__version__ != PARSER_VERSION:
        raise ValueError("installed_parser_version_mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("format") != "sqlrl-m6-08j-frozen-v1" or manifest.get("rule_version") != RULE_VERSION:
        raise ValueError("manifest_version_mismatch")
    if manifest.get("parser_version") != PARSER_VERSION or manifest.get("ast_rule_version") != AST_RULE_VERSION:
        raise ValueError("parser_or_ast_rule_mismatch")
    for path, expected in ((dataset_path, manifest["dataset_sha256"]),
                           (manifest["reviews_path"], manifest["reviews_sha256"]),
                           (manifest["monitor_path"], manifest["monitor_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError("frozen_hash_mismatch:" + str(path))
    rows, reviews = read_jsonl(dataset_path), read_jsonl(manifest["reviews_path"])
    index = {r["trajectory_id"]: r for r in reviews}
    if len(index) != len(reviews):
        raise ValueError("duplicate_reviews")
    hashes = {r["trajectory_id"]: content_hash(r) for r in rows}
    if len(hashes) != len(rows) or hashes != manifest["record_hashes"]:
        raise ValueError("record_hash_mismatch")
    expected_isolation = {str(p): sha256_file(p) for p in isolation_inputs()}
    if manifest.get("isolation_input_hashes") != expected_isolation:
        raise ValueError("split_isolation_evidence_changed")
    errors = quota_errors(rows)
    monitor = {r["id"] for r in read_jsonl(manifest["monitor_path"])}
    if len(monitor) != 64 or monitor & {r["id"] for r in rows}:
        errors.append("monitor_leakage_or_invalid_size")
    errors += check_isolation(rows, monitor)
    for r in rows:
        errors += review_errors(r, index.get(r["trajectory_id"]))
        if r.get("runtime_fingerprint_sha256") != manifest["runtime_fingerprint_sha256"]:
            errors.append("runtime_contract_mismatch")
        if r.get("content_sha256") != content_hash(r):
            errors.append("content_hash_mismatch")
    if errors:
        raise ValueError("training_rejected:" + ",".join(sorted(set(errors))))
    return manifest
