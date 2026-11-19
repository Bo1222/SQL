"""M6-08k adversarial regression tests. Fixtures never grant production reviews."""
import copy
import json
import sqlite3
from pathlib import Path

import pytest

from e1_alignment.common import canonical_hash, read_jsonl, write_jsonl
from e1_alignment.runtime import Runtime, RuntimeContract, process_generation
from e1_alignment.revision.policy import load_policy, policy_hash
from e1_alignment.revision.construct import build, action, score_exact
from e1_alignment.revision.quality import prepare, verified_repair, check, review_errors
from e1_alignment.revision.provenance import pointer, verify_history, history, match_prefix
from e1_alignment.revision.budget import reserve, original_identity
from e1_alignment.revision.selection import features, distribution, quota_errors, feasible, learning_value
from e1_alignment.revision import RULE_VERSION
from e1_alignment.rebuild.quality import REVIEW_FIELDS
from reward.bird_ex_reward import score_sql


class Tokenizer:
    eos_token_id = 999
    def encode(self, text, **kwargs):
        return list(text.encode())
    def decode(self, ids, **kwargs):
        return bytes(i for i in ids if i != self.eos_token_id).decode()


class FixtureRuntime(Runtime):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.contract = RuntimeContract()
        self.lookups = []
        self._tools = {}
    def prompt_for_row(self, row):
        return "Question: " + row["question"]
    def schema_tool(self, db_id):
        def lookup(arguments):
            self.lookups.append(arguments)
            return {"status": "ok", "observation": "Tables:\n- a\n  - x INT\nRelationships:"}
        return lookup


@pytest.fixture
def task(tmp_path):
    path = tmp_path / "a.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE a(x INT)")
        con.execute("INSERT INTO a VALUES (7)")
    return {"id": "train_q", "db_id": "a", "question": "What is x?", "evidence": "", "db_path": str(path), "gold_sql": "SELECT x FROM a"}


def failure(task, sql="SELECT missing FROM a"):
    runtime = FixtureRuntime()
    state = runtime.new_state(task)
    action(state, "sql_execute", sql)
    return runtime, state.record()


def bind(record, original, tmp_path):
    source = tmp_path / "original.jsonl"
    acquired = tmp_path / "acquired.jsonl"
    write_jsonl(source, [original]); write_jsonl(acquired, [record])
    receipt = pointer(acquired, 1, record)
    record["origin"] = record["history_origin"] = pointer(source, 1, original)
    record["acquisition_receipt"] = receipt
    record["trajectory_id"] = "test"
    return record


def test_gold_failure_continuation_keeps_eos_feedback_and_zero_prefix_loss(task, tmp_path):
    runtime, original = failure(task)
    record, attempt = build(task, runtime, score_sql, "fp", original, 1)
    assert record and not attempt["reason"]
    assert record["segments"][1]["ids"][-1] == 999
    assert history(record["segments"][:3]) == history(original["segments"])
    assert record["tool_events"][0] == original["tool_events"][0]
    assert all(not any(s["loss_mask"]) for s in record["segments"][:3])
    assert record["autonomous_repair"] is False
    repair = verified_repair(record, score_sql)
    assert repair["verified"] and repair["error_class"] == "column_alias"
    record.update(has_verified_failure_repair=True, repair_evidence=repair)
    record = prepare(bind(record, original, tmp_path), load_policy())
    result = check(record, load_policy(), {task["id"]: task}, set(), "fp", runtime, set())
    assert result["ok"], result["reasons"]
    assert "gold_first_lookup_not_question_evidence" not in result["reasons"]


@pytest.mark.parametrize("kind", ["lookup_only", "already_correct", "infrastructure", "gold_failure"])
def test_non_failures_never_create_repairs(task, kind):
    runtime = FixtureRuntime()
    state = runtime.new_state(task)
    if kind == "lookup_only":
        action(state, "schema_lookup", {"query": "x"})
    else:
        action(state, "sql_execute", task["gold_sql"] if kind == "already_correct" else "SELECT missing FROM a")
    original = state.record()
    score = score_sql
    if kind == "infrastructure":
        original["tool_events"][0]["result"]["failure_origin"] = "infrastructure"
    if kind == "gold_failure":
        score = lambda *args: {"ok": False, "correct": False, "infrastructure_error": True}
    record, attempted = build(task, runtime, score, "fp", original, 1)
    assert record is None and attempted["reason"]


@pytest.mark.parametrize("fault", ["missing", "negative", "out_of_range", "wrong_sql", "unexecuted_suffix"])
def test_repair_requires_bound_real_execution_receipts(task, fault):
    runtime, original = failure(task)
    record, _ = build(task, runtime, score_sql, "fp", original, 1)
    executes = [s for s in record["segments"] if s.get("action") == "sql_execute"]
    assert verified_repair(record, score_sql)["verified"]
    if fault == "missing":
        executes[0].pop("tool_event_index")
    elif fault == "negative":
        executes[0]["tool_event_index"] = -1
    elif fault == "out_of_range":
        executes[0]["tool_event_index"] = len(record["tool_events"])
    elif fault == "wrong_sql":
        record["tool_events"][executes[0]["tool_event_index"]]["arguments"] = "SELECT 0"
    else:
        executes[1].pop("tool_event_index")
    assert not verified_repair(record, score_sql)["verified"]


def test_history_and_receipts_cannot_be_forged_by_metadata_or_renamed_parent(task, tmp_path):
    runtime, original = failure(task)
    record, _ = build(task, runtime, score_sql, "fp", original, 1)
    bind(record, original, tmp_path)
    verify_history(record)
    missing = {**record, "acquisition_receipt": None}
    with pytest.raises(ValueError, match="requires_real_acquisition_receipt"):
        verify_history(missing)
    changed = copy.deepcopy(record)
    changed["segments"][1]["ids"] = changed["segments"][1]["ids"][:-1]
    with pytest.raises(ValueError, match="historical_prefix_changed"):
        verify_history(changed)
    changed = copy.deepcopy(record)
    changed["segments"][2]["text"] += " made up column"
    with pytest.raises(ValueError, match="historical_prefix_changed"):
        verify_history(changed)
    changed = copy.deepcopy(record)
    changed["segments"][-1]["arguments"] = "SELECT 0"
    with pytest.raises(ValueError, match="acquired_history_mismatch"):
        verify_history(changed)
    assert original_identity(record) == original_identity({**record, "trajectory_id": "renamed", "parent_version_id": "renamed"})
    assert original_identity(record) == original_identity({**record, "origin": {**record["origin"], "file": "renamed-parent"}})


def test_budget_reservations_survive_failed_requests_and_policy_changes(tmp_path):
    p = {**load_policy(), "gold_attempt_cap": 1, "model_state_cap": 1}
    path = tmp_path / "quota.jsonl"
    reserve(path, "gold", "original", p)
    with pytest.raises(ValueError, match="already_reserved"):
        reserve(path, "gold", "original", p)
    with pytest.raises(ValueError, match="quota_exhausted"):
        reserve(path, "gold", "another", p)
    got = reserve(path, "model", "state", p)
    assert got["reserved_candidates"] == 2
    with pytest.raises(ValueError, match="quota_exhausted"):
        reserve(path, "model", "new_state", p)
    assert len(read_jsonl(path)) == 2


def test_execution_cache_preserves_literal_spaces_and_caches_no_infrastructure(monkeypatch, tmp_path):
    import e1_alignment.revision.scoring as cache
    database = str((tmp_path / "a.sqlite").resolve())
    monkeypatch.setattr(cache, "_hashes", {database: "verified-db-hash"})
    monkeypatch.setattr(cache, "_loaded", True)
    monkeypatch.setattr(cache, "_futures", {})
    monkeypatch.setattr(cache, "CACHE", tmp_path / "scores.jsonl")
    calls = []
    def execute(db, sql, gold, **kwargs):
        calls.append((sql, gold))
        return {"ok": True, "correct": sql == gold, "infrastructure_error": sql == "infra"}
    monkeypatch.setattr(cache, "execute_score", execute)
    assert cache.score_sql(database, "SELECT 'x  y'", "SELECT 'x  y'")["correct"]
    assert not cache.score_sql(database, "SELECT 'x y'", "SELECT 'x  y'")["correct"]
    cache.score_sql(database, "SELECT 'x  y'", "SELECT 'x  y'")
    assert len(calls) == 2
    cache.score_sql(database, "infra", "gold"); cache.score_sql(database, "infra", "gold")
    assert len(calls) == 4


def test_inference_watchdog_requires_actual_tokens_not_empty_completions(monkeypatch):
    from e1_alignment.revision.sample import ProgressEngine
    from contextlib import nullcontext
    import e1_alignment.revision.sample as sample
    current = [0.0]
    monkeypatch.setattr(sample.time, "monotonic", lambda: current[0])
    monkeypatch.setattr(sample, "health_watchdog", lambda _: nullcontext())
    class Empty:
        def generate(self, *args, **kwargs): return [[]]
    engine = ProgressEngine(Empty(), 900)
    current[0] = 899
    assert engine.generate([[1]], [2]) == [[]]
    current[0] = 901
    with pytest.raises(TimeoutError, match="no_progress"):
        engine.generate([[1]], [2])


def synthetic_row(index, repair=False):
    initial = "schema_lookup" if index % 2 else "sql_execute"
    actions = [{"kind": "action", "action": "sql_execute" if repair else initial, "ids": [2] * 10,
                "loss_mask": [0] * 10 if repair else [1] * 10}]
    if repair:
        actions.append({"kind": "action", "action": "sql_execute", "ids": [3] * 10, "loss_mask": [1] * 10})
    row = {"id": f"q{index}", "trajectory_id": f"t{index}", "db_id": f"db{index % 20}",
           "construction_source": "gold_assisted" if repair else "model_generated", "start_mode": "failure_continuation" if repair else "question_start",
           "has_verified_failure_repair": repair, "repair_evidence": {"verified": repair, "failure_action_count": 1,
              "error_class": ["column_alias", "table_join", "syntax", "semantic"][index % 4]},
           "segments": [{"kind": "prompt", "ids": [1], "loss_mask": [0]}] + actions}
    row["loss_mask"] = [m for s in row["segments"] for m in s["loss_mask"]]
    return row


def test_selector_minimizes_size_and_counts_unique_question_error_coverage():
    rows = [synthetic_row(i, i < 60) for i in range(220)]
    selected, report = feasible(rows, load_policy())
    assert report["status"] == "selected", report
    assert len(selected) == 200 and not quota_errors(selected, load_policy())
    d = distribution(selected)
    assert d["repair_questions"] >= 40
    assert all(d["repair_questions_by_error"][e] >= 5 for e in load_policy()["repair_errors"])


def test_selector_prefers_independent_questions_before_second_trajectory():
    rows = [synthetic_row(i, i < 60) for i in range(200)]
    for i in range(100, 120):
        duplicate = synthetic_row(i, True)
        duplicate["trajectory_id"] += "_different_repair_value"
        rows.append(duplicate)
    selected, report = feasible(rows, load_policy())
    assert report["status"] == "selected"
    assert len(selected) == 200 and len({r["id"] for r in selected}) == 200


def test_source_label_and_aliases_do_not_create_distinct_learning_value():
    r = synthetic_row(0)
    assert learning_value(r) == learning_value({**r, "construction_source": "gold_assisted"})
    duplicate = {**r, "trajectory_id": "second"}
    assert "duplicate_learning_value" in quota_errors([r, duplicate], load_policy())
    repair = synthetic_row(1, True)
    assert learning_value(repair) == learning_value({**repair, "start_mode": "question_start"})


def test_continuations_never_count_initial_and_history_never_counts_repair_tokens():
    row = synthetic_row(0, True)
    row["segments"][1]["loss_mask"] = [0] * 1000
    assert features(row)["initial"] is None and features(row)["repair_tokens"] == 10
    assert distribution([row])["initial_records"] == 0
    row["has_verified_failure_repair"] = False
    assert features(row)["repair"] == 0


def test_no_review_and_changed_program_result_block_loading():
    from tests.test_sft_rebuild import reviewed_row
    row, review = reviewed_row()
    row["quality_rule_version"] = RULE_VERSION
    row["program_check"]["rule_version"] = RULE_VERSION
    review["rule_version"] = RULE_VERSION
    review["data_policy_sha256"] = policy_hash(load_policy())
    review["program_check_sha256"] = canonical_hash(row["program_check"])
    from e1_alignment.rebuild.quality import content_hash
    row["content_sha256"] = content_hash(row)
    row["program_check"]["content_sha256"] = row["content_sha256"]
    review["content_sha256"] = row["content_sha256"]
    review["program_check_sha256"] = canonical_hash(row["program_check"])
    assert not review_errors(row, review, load_policy())
    assert review_errors(row, None, load_policy())
    row["program_check"]["warnings"] = ["new_review_needed"]
    assert "review_program_check_changed" in review_errors(row, review, load_policy())


def test_data_only_entry_does_not_import_or_invoke_training_or_evaluation():
    import ast
    for path in Path("e1_alignment/revision").glob("*.py"):
        code = ast.parse(path.read_text())
        for node in ast.walk(code):
            if isinstance(node, ast.ImportFrom):
                assert not any(s in (node.module or "") for s in ("sft_train", "rebuild.workflow", "evaluate_agent", "rebuild.assess"))
    from e1_alignment.rebuild.freeze import verify_frozen
    # Version dispatch is separately exercised without constructing any model.
    assert callable(verify_frozen)


def test_versioned_freeze_dispatch_offline_load_and_hash_failure(task, tmp_path, monkeypatch):
    from e1_alignment.revision.freeze import freeze, offline_load, code_hashes
    from e1_alignment.rebuild.freeze import verify_frozen
    from e1_alignment.common import write_json
    from e1_alignment.rebuild.quality import content_hash
    runtime = FixtureRuntime()
    record, _ = build(task, runtime, score_sql, "fp")
    p = {**load_policy(), "min_records": 1, "max_records": 1, "min_databases": 1,
         "gold_record_max": 1, "gold_token_max": 1, "repair_record_min": 0,
         "repair_token_min": 0, "min_repair_questions": 0, "min_questions_per_error": 0,
         "initial_execute_min": 0}
    monitor = tmp_path / "monitor.jsonl"
    write_jsonl(monitor, [{"id": f"monitor{i}"} for i in range(64)])
    p["monitor_path"] = str(monitor)
    record = bind(record, task, tmp_path)
    record.update(trajectory_id="fixture",
                  has_verified_failure_repair=False, repair_evidence={"verified": False})
    record = prepare(record, p)
    record["program_check"] = check(record, p, {task["id"]: task}, set(), "fp", runtime, set())
    assert record["program_check"]["ok"], record["program_check"]["reasons"]
    review = {"id": task["id"], "trajectory_id": "fixture", "content_sha256": content_hash(record),
              "rule_version": RULE_VERSION, "data_policy_sha256": policy_hash(p),
              "program_check_sha256": canonical_hash(record["program_check"]), "status": "pass",
              "reviewer": "regression-fixture-only", "method": "individual_assistant_review",
              "reviewed_at": "2026-11-10", "review_seconds": 1,
              **{f: {"verdict": "pass", "notes": "Fixture: x=7, one visible column, no join or failure, correct masks."} for f in REVIEW_FIELDS}}
    review_path = tmp_path / "reviews.jsonl"
    write_jsonl(review_path, [review])
    policy_path = tmp_path / "policy.json"; write_json(policy_path, p)
    pref = tmp_path / "preflight.json"
    write_json(pref, {"code_hashes": code_hashes(), "data_policy_sha256": policy_hash(p)})
    runtime_file = tmp_path / "runtime.json"; write_json(runtime_file, {"fingerprint": "fp"})
    p["runtime_path"] = str(runtime_file)
    record = prepare(record, p)
    record["program_check"] = check(record, p, {task["id"]: task}, set(), "fp", runtime, set())
    review.update(content_sha256=content_hash(record), data_policy_sha256=policy_hash(p),
                  program_check_sha256=canonical_hash(record["program_check"]))
    write_jsonl(review_path, [review]); write_json(policy_path, p)
    write_json(pref, {"code_hashes": code_hashes(), "data_policy_sha256": policy_hash(p)})
    monkeypatch.setattr("e1_alignment.revision.freeze.check_isolation", lambda *args: [])
    dataset = tmp_path / "train.jsonl"; manifest = tmp_path / "manifest.json"
    freeze([record], review_path, policy_path, pref, dataset, manifest, "fp")
    assert verify_frozen(dataset, manifest)["status"] == "ready_for_sft_data"
    assert offline_load(dataset, manifest)["model_started"] is False
    write_jsonl(review_path, [{**review, "status": "pending"}])
    with pytest.raises(ValueError, match="frozen_hash_mismatch"):
        offline_load(dataset, manifest)
