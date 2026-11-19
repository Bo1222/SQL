"""Policy/version-aware data freezing and offline loading, with fail-closed review."""
import json
from pathlib import Path
from e1_alignment.common import read_jsonl, write_json, write_jsonl, sha256_file, canonical_hash
from e1_alignment.rebuild.ast_check import PARSER_VERSION, AST_RULE_VERSION
from e1_alignment.rebuild.quality import content_hash
from e1_alignment.rebuild.freeze import check_isolation, isolation_inputs
from e1_alignment.revision import FORMAT, RULE_VERSION
from e1_alignment.revision.policy import load_policy, policy_hash
from e1_alignment.revision.quality import review_errors
from e1_alignment.revision.provenance import verify_history
from e1_alignment.revision.selection import quota_errors, distribution


def code_hashes():
    paths = sorted(Path("e1_alignment/revision").glob("*.py"))
    paths += [Path("e1_alignment/rebuild/quality.py"), Path("e1_alignment/rebuild/ast_check.py"),
              Path("e1_alignment/rebuild/visibility.py"), Path("e1_alignment/dataset.py")]
    return {str(p.resolve()): sha256_file(p) for p in paths}


def validate(rows, reviews, policy, fingerprint):
    index = {r["trajectory_id"]: r for r in reviews}
    errors = quota_errors(rows, policy)
    if len(index) != len(reviews):
        errors.append("duplicate_reviews")
    quarantined = {r["id"] for r in reviews if r.get("quarantine_question")}
    monitor = read_jsonl(policy["monitor_path"])
    if len(monitor) != 64 or len({r["id"] for r in monitor}) != 64:
        errors.append("invalid_monitor_set")
    errors += check_isolation(rows, {r["id"] for r in monitor})
    if len({r["trajectory_id"] for r in rows}) != len(rows):
        errors.append("duplicate_trajectory")
    for row in rows:
        errors += review_errors(row, index.get(row["trajectory_id"]), policy)
        if row["id"] in quarantined:
            errors.append("question_annotation_quarantined")
        if row.get("data_policy_sha256") != policy_hash(policy) or row.get("runtime_fingerprint_sha256") != fingerprint:
            errors.append("record_policy_or_runtime_changed")
        if row.get("content_sha256") != content_hash(row):
            errors.append("record_content_changed")
        try:
            verify_history(row)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            errors.append("source_changed:" + str(exc))
        ids = [t for s in row["segments"] for t in s["ids"]]
        masks = [m for s in row["segments"] for m in s["loss_mask"]]
        if ids != row["input_ids"] or masks != row["loss_mask"] or len(ids) != len(masks) or not ids or masks[0] or not any(masks[1:]):
            errors.append("offline_input_invalid")
    if errors:
        raise ValueError("data_loading_rejected:" + ",".join(sorted(set(errors))))


def freeze(rows, reviews_path, policy_path, preflight_path, dataset_path, manifest_path, fingerprint):
    policy = load_policy(policy_path)
    validate(rows, read_jsonl(reviews_path), policy, fingerprint)
    preflight = json.loads(Path(preflight_path).read_text())
    if preflight["code_hashes"] != code_hashes() or preflight["data_policy_sha256"] != policy_hash(policy):
        raise ValueError("preflight_code_or_policy_changed")
    if Path(dataset_path).exists() or Path(manifest_path).exists():
        raise FileExistsError("frozen_artifacts_are_immutable")
    write_jsonl(dataset_path, rows)
    bound = [reviews_path, policy_path, preflight_path, policy["monitor_path"], policy["runtime_path"]]
    bound += list(isolation_inputs())
    branch = Path(dataset_path).parent
    bound += [path for name in ("acquisition_ledger.jsonl", "gold_attempts.jsonl", "source_model.raw.jsonl",
                               "model_candidates.jsonl", "revisions.jsonl")
              if (path := branch / name).is_file()]
    for row in rows:
        bound += [row["origin"]["file"], row["history_origin"]["file"]]
        if row.get("acquisition_receipt"):
            bound.append(row["acquisition_receipt"]["file"])
    manifest = {"format": FORMAT, "status": "ready_for_sft_data", "rule_version": RULE_VERSION,
                "parser_version": PARSER_VERSION, "ast_rule_version": AST_RULE_VERSION,
                "data_policy_sha256": policy_hash(policy), "policy_path": str(Path(policy_path).resolve()),
                "reviews_path": str(Path(reviews_path).resolve()), "runtime_fingerprint_sha256": fingerprint,
                "monitor_path": str(Path(policy["monitor_path"]).resolve()),
                "reviews_sha256": sha256_file(reviews_path), "monitor_sha256": sha256_file(policy["monitor_path"]),
                "dataset_sha256": sha256_file(dataset_path), "code_hashes": code_hashes(),
                "bound_files": {str(Path(p).resolve()): sha256_file(p) for p in bound},
                "record_hashes": {r["trajectory_id"]: content_hash(r) for r in rows}, "distribution": distribution(rows)}
    write_json(manifest_path, manifest)
    return manifest


def verify_frozen(dataset_path, manifest_path):
    import sqlglot
    m = json.loads(Path(manifest_path).read_text())
    if m.get("format") != FORMAT or m.get("status") != "ready_for_sft_data" or m.get("rule_version") != RULE_VERSION:
        raise ValueError("manifest_version_mismatch")
    if sqlglot.__version__ != PARSER_VERSION or m.get("parser_version") != PARSER_VERSION or m.get("ast_rule_version") != AST_RULE_VERSION:
        raise ValueError("parser_rule_mismatch")
    if m.get("code_hashes") != code_hashes():
        raise ValueError("frozen_validation_code_changed")
    for p, sha in {**m["bound_files"], str(dataset_path): m["dataset_sha256"]}.items():
        if sha256_file(p) != sha:
            raise ValueError("frozen_hash_mismatch:" + p)
    policy = load_policy(m["policy_path"])
    if policy_hash(policy) != m["data_policy_sha256"]:
        raise ValueError("policy_changed")
    rows = read_jsonl(dataset_path)
    if {r["trajectory_id"]: content_hash(r) for r in rows} != m["record_hashes"]:
        raise ValueError("record_hash_mismatch")
    validate(rows, read_jsonl(m["reviews_path"]), policy, m["runtime_fingerprint_sha256"])
    return m


def offline_load(dataset_path, manifest_path):
    manifest = verify_frozen(dataset_path, manifest_path)
    # Same input/label convention as sft_train.ExampleDataset, without importing a model.
    rows = read_jsonl(dataset_path)
    for row in rows:
        if len(row["input_ids"]) > 6144 or any(m not in (0, 1) for m in row["loss_mask"]):
            raise ValueError("offline_context_or_mask_invalid")
    return {"status": "offline_load_passed", "records": len(rows), "dataset_sha256": manifest["dataset_sha256"],
            "model_started": False, "optimizer_started": False}
