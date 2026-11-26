"""CPU tests for v3 prompt wiring, the seed path, and the forced-lookup probe."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import (  # noqa: E402
    PROMPT_VERSION,
    PROMPT_VERSION_V3,
    build_prompt,
    prompt_version_for,
)
from agent.runtime_contract import (  # noqa: E402
    CONTRACT_ID,
    CONTRACT_V1,
    CONTRACT_V3,
    FINGERPRINT,
    FINGERPRINT_V1,
    FINGERPRINT_V3,
    config as runtime_config,
    contract_for,
    prompt_for,
    validate_extra,
)
from agent.sql_episode import EpisodeConfig, episode_to_record, run_episode  # noqa: E402
from eval.evaluate_agent import build_eval_prompt  # noqa: E402
from scripts.probe_forced_exact_lookup import (  # noqa: E402
    cmd_summarize,
    mcnemar_one_sided,
    select_seed,
)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8")


class FakeStore:
    table_names = ["t", "u"]


def test_v3_prompt_has_markers_and_differs_from_v1():
    catalog = "t\nu"
    v1 = build_prompt("q", "e", catalog, max_tools=4, prompt_mode="query")
    v3 = build_prompt("q", "e", catalog, max_tools=4, prompt_mode="query_v3")
    assert "catalog-three-action-query-v3" in v3
    assert '<schema_lookup>{"table"' in v3
    assert "catalog-three-action-query-v3" not in v1
    assert v1 != v3
    assert prompt_version_for("query") == PROMPT_VERSION
    assert prompt_version_for("query_v3") == PROMPT_VERSION_V3


def test_build_eval_prompt_contract_wins_and_prompt_reaches_engine_input():
    metadata = {"tables": [{"name": "t"}]}
    args = SimpleNamespace(prompt_mode="query", max_tools=4, chat_template=False)
    prompt, meta = build_eval_prompt(
        {"question": "q", "evidence": "e"}, metadata, args, CharTokenizer(),
        contract_id="A-e1.2-typed-v3")
    assert "catalog-three-action-query-v3" in prompt
    assert '<schema_lookup>{"table"' in prompt
    assert meta["prompt_version"] == PROMPT_VERSION_V3
    assert meta["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert prompt.startswith(meta["prompt_prefix"])
    seen = []

    def fake_engine(ids, max_new_tokens):
        seen.append(list(ids))
        return list("<answer>SELECT 1</answer>".encode())

    tokenizer = CharTokenizer()
    state = run_episode(
        {"question": "q", "evidence": "e", "catalog_tables": ["t"]},
        tokenizer, fake_engine,
        lambda seed: {"status": "ok", "observation": "T t(y TEXT)"},
        lambda sql: {"status": "ok", "observation": "1"},
        EpisodeConfig(max_rounds=2, max_context=8192, max_observation_tokens=1024),
        prompt=prompt)
    assert state["segments"][0]["text"] == prompt
    assert "catalog-three-action-query-v3" in tokenizer.decode(seen[0])
    assert state["finish_reason"] == "answer_submitted"


def test_registry_v1_aliases_and_v3_fingerprint():
    assert CONTRACT_ID == "A-e1.2-typed-v1"
    assert FINGERPRINT == FINGERPRINT_V1
    assert FINGERPRINT_V1 == hashlib.sha256(
        json.dumps(CONTRACT_V1, sort_keys=True).encode()).hexdigest()
    assert FINGERPRINT_V3 == hashlib.sha256(
        json.dumps(CONTRACT_V3, sort_keys=True).encode()).hexdigest()
    assert contract_for("A-e1.2-typed-v3")["prompt_mode"] == "query_v3"
    assert runtime_config("A-e1.2-typed-v3").observation_mode == "compact_v2"
    assert runtime_config("A-e1.2-typed-v3").max_observation_tokens == 1024
    assert prompt_for({"question": "q", "evidence": "e"}, FakeStore(),
                      contract_id="A-e1.2-typed-v3").startswith(
        "You are a read-only SQLite database agent")


def test_validate_extra_v3_and_mismatch():
    extra = {
        "runtime_contract": "A-e1.2-typed-v3",
        "runtime_fingerprint": FINGERPRINT_V3,
        "score_mode": "bird_set", "retrieval_version": "E1.2",
        "observation_mode": "compact_v2", "reward_contract": "typed-v1",
        "prompt_mode": "query_v3", "prompt_version": PROMPT_VERSION_V3,
    }
    validate_extra(extra, contract_id="A-e1.2-typed-v3")
    extra["prompt_mode"] = "query"
    try:
        validate_extra(extra, contract_id="A-e1.2-typed-v3")
    except ValueError as exc:
        assert "prompt_mode" in str(exc)
    else:
        raise AssertionError("expected prompt_mode mismatch")


def test_select_seed_cross_table_and_ties():
    catalog = {"a": ["y", "x"], "b": ["z"]}
    requirements = {"ok": True, "tables": ["a", "b"],
                    "columns": ["a.x", "a.y", "b.z"]}
    seed, info = select_seed(requirements, catalog)
    assert seed == {"query": "a", "table": "a", "column": "y"}
    assert info["skipped_reason"] is None
    assert info["n_required_columns"] == 3

    tie_catalog = {"b": ["z"], "a": ["x"]}
    tie_requirements = {"ok": True, "tables": ["a", "b"],
                        "columns": ["a.x", "b.z"]}
    seed, info = select_seed(tie_requirements, tie_catalog)
    assert seed["table"] == "a"
    assert seed["column"] == "x"


def test_select_seed_skips_are_reported():
    cases = [
        ({"ok": False, "tables": [], "columns": []}, {"a": ["x"]},
         "requirements_not_ok"),
        ({"ok": True, "tables": ["a"], "columns": []}, {"a": ["x"]},
         "no_required_columns"),
        ({"ok": True, "tables": ["c"], "columns": ["c.x"]}, {"a": ["x"]},
         "no_required_table_in_catalog"),
        ({"ok": True, "tables": ["a"], "columns": ["a.nope"]}, {"a": ["x"]},
         "required_column_absent_from_catalog"),
    ]
    for requirements, catalog, reason in cases:
        seed, info = select_seed(requirements, catalog)
        assert seed is None
        assert info["skipped_reason"] == reason


def test_seed_path_counts_and_both_flags_rejected():
    tokenizer = CharTokenizer()
    schema_tool = lambda seed: {"status": "ok", "observation": "T t(y TEXT)"}
    sql_tool = lambda sql: {"status": "ok", "observation": "1"}
    generate = lambda ids, n: list("<answer>SELECT 1</answer>".encode())
    config = EpisodeConfig(max_rounds=2, max_context=8192,
                           max_observation_tokens=1024)
    state = run_episode({"question": "q", "catalog_tables": ["t"]}, tokenizer,
                        generate, schema_tool, sql_tool, config,
                        seed_lookup={"table": "t", "column": "y"})
    assert state["finish_reason"] == "answer_submitted"
    assert state["tool_calls"] == 1 and state["schema_lookups"] == 1
    assert state["seeded_lookup"] == {"table": "t", "column": "y"}
    assert state["tool_events"][0]["seeded"] is True
    record = episode_to_record(state, {"id": "q"})
    assert record["seeded_lookup"] == {"table": "t", "column": "y"}

    try:
        run_episode({"question": "q", "catalog_tables": ["t"]}, tokenizer,
                    generate, schema_tool, sql_tool,
                    EpisodeConfig(force_first_schema_lookup=True),
                    seed_lookup={"table": "t", "column": "y"})
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually exclusive error")


def test_seed_observation_overflow_is_graceful():
    tokenizer = CharTokenizer()
    config = EpisodeConfig(max_rounds=2, max_context=8192,
                           max_observation_tokens=1024, max_response_tokens=5)
    state = run_episode(
        {"question": "q", "catalog_tables": ["t"]}, tokenizer,
        lambda ids, n: list("<answer>SELECT 1</answer>".encode()),
        lambda seed: {"status": "ok", "observation": "a long observation"},
        lambda sql: {"status": "ok", "observation": "1"}, config,
        seed_lookup={"table": "t", "column": "y"})
    assert state["finish_reason"] == "seed_observation_overflow"
    assert state["tool_calls"] == 0


def test_mcnemar_one_sided_and_summarize(tmp_path):
    assert mcnemar_one_sided(4, 0) == 1 / 16
    assert mcnemar_one_sided(4, 1) == 6 / 32
    out = tmp_path
    seed = {"query": "t", "table": "t", "column": "y"}
    base = {
        "finish_reason": "answer_submitted", "seed": seed,
        "tool_events": [{"seeded": True, "arguments": seed,
                         "result": {"status": "ok", "retrieval_metadata": {
                             "inspection": True, "selected_columns": 1,
                             "matched_columns": [{"table": "t", "column": "y"}]}}}],
        "segments": [{"kind": "observation", "text": "exact"}],
    }
    e12 = [
        {**base, "id": "q1", "correct": True, "final_sql": "SELECT y FROM t"},
        {**base, "id": "q2", "correct": False, "final_sql": "SELECT x FROM t"},
    ]
    e11 = [
        {**base, "id": "q1", "correct": False, "final_sql": "SELECT x FROM t",
         "segments": [{"kind": "observation", "text": "query"}],
         "tool_events": [{"seeded": True, "arguments": seed,
                          "result": {"status": "ok", "retrieval_metadata": {
                              "matched_columns": [{"table": "t", "column": "x"}]}}}]},
        {**base, "id": "q2", "correct": False, "final_sql": "SELECT x FROM t",
         "segments": [{"kind": "observation", "text": "query"}],
         "tool_events": [{"seeded": True, "arguments": seed,
                          "result": {"status": "ok", "retrieval_metadata": {
                              "matched_columns": [{"table": "t", "column": "x"}]}}}]},
    ]
    (out / "records_E12_exact.jsonl").write_text(
        "\n".join(json.dumps(row) for row in e12) + "\n")
    (out / "records_E11_ignored.jsonl").write_text(
        "\n".join(json.dumps(row) for row in e11) + "\n")
    args = argparse.Namespace(out=str(out), arms="E12_exact,E11_ignored",
                              baseline_records="")
    summary = cmd_summarize(args)
    assert summary["n_pairs"] == 2
    assert summary["discordant"] == {"e12_only_correct": 1, "e11_only_correct": 0,
                                     "net": 1}
    assert summary["observation_diff"] == 2
    assert summary["sql_diff"] == 1
    assert summary["e12_inspection_rate"] == 1.0
    assert summary["gate"]["n_pairs_200"] is False
    assert summary["gate"]["e11_ignored_rate"] == 1.0
    assert summary["gate"]["pass"] is False
