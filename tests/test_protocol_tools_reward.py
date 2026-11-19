"""Protocol, tools, reward and response-mask tests (no model required)."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import build_prompt, parse_action  # noqa: E402
from agent.sql_episode import EpisodeConfig, encode_observation, run_episode  # noqa: E402
from reward.bird_ex_reward import compute_score, extract_final_sql  # noqa: E402
from tools.sql_execute import SQLExecuteTool, run_sql  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "demo.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_name TEXT)")
    con.executemany("INSERT INTO students VALUES (?,?,?)", [(1, 10, "ann"), (2, 20, "bob")])
    con.executemany("INSERT INTO departments VALUES (?,?)", [(10, "math"), (20, "physics")])
    con.commit()
    con.close()
    return path


def test_parse_action_accepts_each_tool():
    assert parse_action('<schema_lookup>{"query": "students"}</schema_lookup>').name == "schema_lookup"
    assert parse_action("<sql_execute>SELECT 1</sql_execute>").name == "sql_execute"
    assert parse_action("<answer>SELECT 2</answer>").arguments == "SELECT 2"


def test_parse_action_rejects_extra_text_or_multiple_actions():
    for bad in ["prefix <answer>SELECT 1</answer>", "<answer>SELECT 1</answer> tail",
                "<answer>SELECT 1</answer><answer>SELECT 2</answer>",
                "<sql_execute>DROP TABLE students</sql_execute>",
                '<schema_lookup>{"table": "x", "bogus": 1}</schema_lookup>']:
        with pytest.raises(ValueError):
            parse_action(bad)


def test_prompt_is_catalog_only():
    prompt = build_prompt("q?", "ev", "students\ndepartments")
    assert "CREATE TABLE" not in prompt.upper()
    assert "students" in prompt
    assert "dept_name" not in prompt


def test_prompt_modes_match_contracts():
    query = build_prompt("q?", "ev", "students\ndepartments")
    sft = build_prompt("q?", "ev", "students\ndepartments", prompt_mode="sft")
    assert '{"query"' in query and '{"table"' not in query
    assert '{"table"' in sft and "Limits:" in sft
    assert "case-sensitive" not in query.lower()


def test_sql_execute_ok_and_error_classification(db):
    tool = SQLExecuteTool(db)
    ok = tool("SELECT name FROM students ORDER BY id")
    assert ok["status"] == "ok" and ok["rows_preview"][0][0] == "ann"
    missing = tool("SELECT nope FROM students")
    assert missing["status"] == "error" and missing["error_type"] == "schema_error"
    syntax = tool("SELECT FROM")
    assert syntax["error_type"] == "syntax_error"


def test_sql_worker_is_read_only(db):
    result = run_sql(db, "DELETE FROM students")
    assert not result["ok"]


def test_reward_matches_bird_semantics(db):
    gold = "SELECT dept_name FROM departments WHERE id = 10"
    assert compute_score("bird_sql", "<answer>" + gold + "</answer>", gold,
                         {"db_path": str(db)}) == 1.0
    # different SQL, same result -> correct
    assert compute_score("bird_sql", "<answer>SELECT dept_name FROM departments WHERE id IN (10)</answer>",
                         gold, {"db_path": str(db)}) == 1.0
    assert compute_score("bird_sql", "<answer>SELECT dept_name FROM departments</answer>", gold,
                         {"db_path": str(db)}) == 0.0
    assert compute_score("bird_sql", "no answer tag", gold, {"db_path": str(db)}) == 0.0


def test_reward_rejects_duplicate_rows_that_change_count(db):
    gold = "SELECT id FROM students"
    duplicate = "SELECT id FROM students UNION ALL SELECT id FROM students"
    assert compute_score("bird_sql", f"<answer>{duplicate}</answer>", gold, {"db_path": str(db)}) == 0.0


def test_extract_final_sql_uses_last_answer():
    text = "<answer>SELECT 1</answer> noise <answer>SELECT 2</answer>"
    assert extract_final_sql(text) == "SELECT 2"


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 30000 + 1 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i - 1) % 30000) if i else "" for i in ids)


def test_observation_tokens_are_not_policy_targets():
    tokenizer = _FakeTokenizer()
    ids, text = encode_observation(tokenizer, {"status": "ok", "observation": "x" * 50}, 128)
    assert ids and text.startswith("<observation>")
    state = run_episode(
        {"question": "q", "evidence": "", "catalog_tables": ["t"]},
        tokenizer,
        lambda ids, limit: tokenizer.encode("<answer>SELECT 1</answer>"),
        lambda args: {"status": "ok", "observation": "ref"},
        lambda sql: {"status": "ok", "columns": [], "rows_preview": []},
        EpisodeConfig(max_rounds=2),
    )
    kinds = [s["kind"] for s in state["segments"]]
    assert kinds[0] == "prompt" and "action" in kinds
    assert state["finish_reason"] == "answer_submitted"
    assert state["final_sql"] == "SELECT 1"


def test_episode_respects_tool_budget():
    tokenizer = _FakeTokenizer()
    calls = {"n": 0}

    def generate(ids, limit):
        calls["n"] += 1
        return tokenizer.encode("<sql_execute>SELECT 1</sql_execute>")

    state = run_episode({"question": "q", "catalog_tables": ["t"]}, tokenizer, generate,
                        lambda args: {"status": "ok"}, lambda sql: {"status": "ok", "columns": []},
                        EpisodeConfig(max_rounds=6, max_sql_executes=2, max_tools=4))
    assert state["sql_executes"] == 2
    assert state["finish_reason"] in {"tool_limit", "turn_limit"}

