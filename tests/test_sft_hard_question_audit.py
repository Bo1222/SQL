"""CPU tests for Step 0/1 hard-question audit and schema bundles."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_sft_allzero_labels import (  # noqa: E402
    check_evidence_filters,
    check_question,
    classify,
)
from scripts.build_schema_grounding_bundles import build_one  # noqa: E402
from scripts.sft_hard_question_common import (  # noqa: E402
    SchemaCatalog,
    execute_gold,
    gold_join_edges,
    select_columns,
    sql_string_literals,
    text_quoted_literals,
)


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE coins (
            id INTEGER PRIMARY KEY,
            name TEXT,
            symbol TEXT
        );
        CREATE TABLE historical (
            coin_id INTEGER,
            date TEXT,
            circulating_supply REAL
        );
        CREATE TABLE current (
            bioguide_id TEXT PRIMARY KEY,
            official_full_name TEXT,
            birth_place TEXT,
            birth_region TEXT
        );
        CREATE TABLE current_terms (
            bioguide TEXT PRIMARY KEY,
            party TEXT,
            state_rank TEXT
        );
        INSERT INTO coins VALUES (1, 'Bitcoin', 'BTC');
        INSERT INTO historical VALUES (1, '2013-04-28', 100.0);
        INSERT INTO current VALUES ('A1', 'Ada Example', 'California', 'CA');
        INSERT INTO current_terms VALUES ('A1', 'Republican', 'junior');
    """)
    con.close()
    return path


def test_quoted_literals_ignore_possessive_apostrophes():
    text = ("aldermen's full name refers to primary_description = 'ARSON'; "
            "condition described as cystitis refers to conditions.DESCRIPTION = ''Cystitis'")
    values = text_quoted_literals(text)
    assert "ARSON" in values
    assert "Cystitis" in values
    assert not any("full name" in value for value in values)
    assert text_quoted_literals('film titled "BLADE POLISH"') == ["BLADE POLISH"]


def test_sql_literals_decode_doubled_quotes():
    assert sql_string_literals("WHERE x = 'O''Sullivan' AND y = 'a'") == [
        "O'Sullivan", "a"]


def test_catalog_prefix_join_and_column_priority(db_path):
    catalog = SchemaCatalog(db_path)
    edges = catalog.name_match_edges()
    assert any(edge["left_table"] == "historical"
               and edge["left_column"] == "coin_id"
               and edge["right_table"] == "coins"
               and edge["right_column"] == "id"
               and edge["source"] == "fk_name_prefix"
               for edge in edges)
    cols = select_columns(catalog, "historical", "coin supply", "",
                          max_columns=16, peer_tables={"coins", "historical"})
    assert "coin_id" in cols
    cols = select_columns(catalog, "current", "full names", "",
                          max_columns=2, peer_tables={"current", "current_terms"})
    assert "official_full_name" in cols


def test_execute_gold_and_join_extraction(db_path):
    result = execute_gold(db_path, "SELECT name FROM coins")
    assert result["ok"] and result["rows"] == 1 and not result["empty"]
    edges = gold_join_edges(
        "SELECT T1.name FROM coins AS T1 "
        "JOIN historical AS T2 ON T1.id = T2.coin_id",
        {"coins", "historical"})
    assert edges and edges[0]["left_column"] == "id"
    assert edges[0]["right_column"] == "coin_id"


def test_evidence_filter_conflict_and_aggregate(db_path):
    catalog = SchemaCatalog(db_path)
    conflicts, ambiguities = [], []
    check_evidence_filters(
        "born in California refers to birth_place = 'California'",
        "SELECT birth_region FROM current WHERE birth_region = 'California'",
        catalog, conflicts, ambiguities)
    assert any(item["code"] == "evidence_filter_column_not_in_gold"
               for item in conflicts)

    conflicts, ambiguities = [], []
    check_evidence_filters(
        "party = 'Republican'",
        "SELECT * FROM current_terms WHERE party = 'Republican'",
        catalog, conflicts, ambiguities)
    assert not conflicts

    ambiguities = []
    check_question("How many people are there?", "SELECT name FROM current", ambiguities)
    assert any(item["code"] == "aggregate_mismatch" for item in ambiguities)
    ambiguities = []
    check_question("How many people are there?", "SELECT COUNT(*) FROM current", ambiguities)
    assert not ambiguities


def test_classify_priority():
    assert classify({}, [{"code": "missing"}], [{"code": "x"}],
                    [{"code": "y"}]) == "dead"
    assert classify({}, [], [{"code": "x"}], [{"code": "y"}]) == "evidence_conflict"
    assert classify({}, [], [], [{"code": "y"}]) == "ambiguous"
    assert classify({}, [], [], []) == "label_clean"


def test_build_one_separates_retrieval_and_training(db_path):
    record = {
        "id": "q1",
        "db_id": "fixture",
        "question": "What is the full name of the Republican?",
        "evidence": "party = 'Republican'",
        "gold_sql": ("SELECT T1.official_full_name FROM current AS T1 "
                     "INNER JOIN current_terms AS T2 "
                     "ON T1.bioguide_id = T2.bioguide "
                     "WHERE T2.party = 'Republican'"),
        "db_path": str(db_path),
    }
    catalog = SchemaCatalog(db_path)
    retrieval, training, eval_row = build_one(
        record, {"category": "schema_column_grounding"}, catalog,
        top_k=4, max_tables=5, max_columns=8)

    def has_key(value, key):
        if isinstance(value, dict):
            return key in value or any(has_key(v, key) for v in value.values())
        if isinstance(value, list):
            return any(has_key(v, key) for v in value)
        return False

    assert not has_key(retrieval, "gold_sql")
    assert training["training_only"] is True
    assert training["gold"]["sql"] == record["gold_sql"]
    assert eval_row["gold_table_covered"] is True
    assert "official_full_name" in json.dumps(retrieval["schema"])
    assert "bioguide_id" in json.dumps(retrieval["schema"])
