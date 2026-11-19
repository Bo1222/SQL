import sqlite3

from agent.sql_episode import EpisodeConfig
from eval.harness_audit import best_lookup_plan, coverage_of, gold_injection, sqlite_requirements


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode()


def test_sqlite_dependencies_resolve_spaces_aliases_and_subqueries(tmp_path):
    db = tmp_path / "a.sqlite"
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE "school data" ("School Name" TEXT, "Free Meals (%)" REAL)')
    req = sqlite_requirements(db, 'SELECT s."School Name" FROM "school data" s WHERE '
                              's."Free Meals (%)" > (SELECT AVG("Free Meals (%)") FROM "school data")')
    assert req["ok"]
    assert req["columns"] == ["school data.free meals (%)", "school data.school name"]


def test_count_star_still_requires_table_and_bad_sql_is_unresolved(tmp_path):
    db = tmp_path / "a.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE t (x INTEGER)")
    assert sqlite_requirements(db, "SELECT COUNT(*) FROM t")["tables"] == ["t"]
    assert not sqlite_requirements(db, "SELECT missing FROM t")["ok"]


def test_metadata_alone_never_counts_as_visible_schema():
    req = {"ok": True, "tables": ["t"], "columns": ["t.x"], "schema": {"t": {"x": "INT"}}}
    text = '<observation>{"status":"ok","observation":"Tables:\\n- t",' \
           '"retrieval_metadata":{"matched_columns":[{"table":"t","column":"x"}]}}</observation>'
    cov = coverage_of(req, [{"kind": "observation", "text": text}])
    assert cov["missing"]["columns"] == ["t.x"]


def test_lookup_search_combines_two_visible_observations():
    req = {"ok": True, "tables": ["t"], "columns": ["t.x", "t.y"],
           "schema": {"t": {"x": "INT", "y": "INT"}}}
    obs = ['<observation>{"status":"ok","observation":"Tables:\\n- t\\n  - ' +
           col + ' INT"}</observation>' for col in ("x", "y")]
    candidates = [{"query": "x"}, {"query": "y"}]
    assert best_lookup_plan(req, candidates, obs, 2)["all_required_schema_visible"]
    assert not best_lookup_plan(req, candidates, obs, 1)["all_required_schema_visible"]


def test_gold_injection_exercises_real_action_budget():
    row = {"question": "one", "gold_sql": "SELECT 1", "catalog_tables": []}
    assert gold_injection(row, Tokenizer())["submitted_exact_gold"]
    result = gold_injection(row, Tokenizer(), EpisodeConfig(max_action_tokens=5))
    assert not result["submitted_exact_gold"]
    assert result["invalid_actions"] == 2


def test_gold_injection_accepts_parser_normalization_of_outer_whitespace():
    row = {"question": "one", "gold_sql": " SELECT 1 \n", "catalog_tables": []}
    result = gold_injection(row, Tokenizer())
    assert result["submitted_exact_gold"]
    assert result["gold_had_outer_whitespace"]


def test_gold_guided_lookups_use_the_real_episode_budgets():
    row = {"question": "one", "gold_sql": "SELECT 1", "catalog_tables": ["t"]}
    result = gold_injection(row, Tokenizer(), EpisodeConfig(max_action_tokens=512),
        [{"table": "t"}, {"query": "x"}], schema_tool=lambda a: {"status": "ok", "observation": "t.x"})
    assert result["submitted_exact_gold"]
    assert result["executed_lookups"] == 2


def test_public_requirements_include_quoted_columns(tmp_path):
    from eval.gold_extraction import extract_requirements
    db = tmp_path / "a.sqlite"
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE t ("Free Meals (%)" REAL)')
    req = extract_requirements('SELECT "Free Meals (%)" FROM t', db)
    assert req["resolved"]
    assert req["columns"] == ["t.free meals (%)"]


def test_oracle_context_uses_catalog_evidence_not_gold_sql_or_literals():
    from scripts.probe_harness_oracle import oracle_context

    class Store:
        catalog = [{"table": "t", "column": "x", "data_type": "TEXT", "description": "entity"},
                   {"table": "other", "column": "secret", "data_type": "TEXT"}]
        relationships = []

    context = oracle_context({"tables": ["t"], "columns": ["t.x"]}, Store())
    assert "entity" in context
    assert "other" not in context
    assert "SELECT" not in context


def test_scoring_diagnostic_exposes_duplicate_row_contract_difference(tmp_path):
    from tools.sql_worker import run
    db = tmp_path / "a.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE t (x INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", [(1,), (1,)])
    result = run({"db_path": str(db), "sql": "SELECT x FROM t", "gold_sql": "SELECT DISTINCT x FROM t",
                  "operation": "score_diagnostic", "timeout": 10})
    assert result["ok"]
    assert not result["project_strict_correct"]
    assert result["bird_set_only_correct"]


def test_official_score_mode_is_explicit_and_defaults_remain_historical(tmp_path):
    from reward.bird_ex_reward import score_sql, compute_score
    db = tmp_path / "a.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE t (x INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", [(1,), (1,)])
    assert not score_sql(db, "SELECT x FROM t", "SELECT DISTINCT x FROM t")["correct"]
    official = score_sql(db, "SELECT x FROM t", "SELECT DISTINCT x FROM t", score_mode="bird_set")
    assert official["correct"]
    assert official["score_implementation_version"] == "bird-official-set-v1"
    assert compute_score("bird", "<answer>SELECT x FROM t</answer>", "SELECT DISTINCT x FROM t",
                         {"db_path": str(db), "score_mode": "bird_set"}) == 1.0


def test_join_recall_does_not_swap_columns_without_swapping_tables():
    from eval.evaluate_agent import recall_from_events
    event = {"action": "schema_lookup", "result": {"retrieval_metadata": {
        "relationships": [{"from_table": "a", "from_column": "y", "to_table": "b", "to_column": "x"}]}}}
    req = {"tables": ["a", "b"], "columns": ["a.x", "b.y"], "join_keys": ["a.x=b.y"]}
    assert recall_from_events([event], req)["join_recall"] == 0.0
