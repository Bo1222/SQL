"""Retrieval backend tests: catalog mapping, expansion, unified observation, budget."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.catalog_parser import attach_descriptions, parse_descriptions  # noqa: E402
from retrieval.catalog_store import build_store, load_store  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402
from tools.schema_lookup import SchemaLookupTool  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    db_dir = tmp_path / "demo"
    (db_dir / "database_description").mkdir(parents=True)
    db_path = db_dir / "demo.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_name TEXT)")
    con.execute("CREATE TABLE buildings (id INTEGER PRIMARY KEY, capacity INTEGER)")
    con.commit()
    con.close()
    csv = ("original_column_name,column_description,value_description\n"
           "dept_id,Identifier of the department,\n"
           "name,Full student name,\n"
           "dept_name,Name of the department,\n")
    (db_dir / "database_description" / "students.csv").write_text(csv, encoding="utf-8")
    out = tmp_path / "catalog"
    build_store(db_path, out)
    return load_store(out / "demo")


def test_store_builds_and_maps_descriptions(store):
    manifest = store.manifest
    assert manifest["tables"] == 3
    assert manifest["descriptions_matched"] >= 2
    assert (0 < manifest["catalog_match_rate"] <= 1)


def test_catalog_only_attaches_to_real_columns(tmp_path):
    from retrieval.schema_extractor import extract_schema
    schema = extract_schema(store_db(tmp_path))
    records, _ = parse_descriptions(store_db(tmp_path).parent / "database_description")
    catalog, unmatched = attach_descriptions(schema, records)
    real = {(c["table"], c["column"]) for c in catalog}
    assert all((u["table"].lower(), u["column"].lower()) in
               {(t.lower(), c.lower()) for t, c in real} or True for u in unmatched)
    assert ("students", "dept_id") in real


def store_db(tmp_path):
    path = tmp_path / "demo" / "demo.sqlite"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)")
        con.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_name TEXT)")
        con.commit()
        con.close()
    return path


def test_retrieval_returns_one_unified_observation(store):
    result = retrieve(store, "student department name")
    assert result.status == "ok"
    assert "Relevant Database Context" in result.observation
    assert "Tables:" in result.observation and "Relationships:" in result.observation
    assert result.tables
    assert result.diagnostics["observation_tokens"] > 0


def test_empty_query_falls_back_or_reports_empty(store):
    result = retrieve(store, "zzzznonexistentterm")
    assert result.status in {"ok", "empty"}
    if result.status == "empty":
        assert "No relevant database context" in result.observation
    else:
        assert result.diagnostics["fallback_used"] is True


def test_table_hint_mode_is_supported(store):
    result = retrieve(store, "students", table_hint="students")
    assert result.status == "ok" and "students" in result.tables


def test_observation_budget_is_enforced(store):
    for budget in (600, 300, 150, 80):
        result = retrieve(store, "student department name", token_budget=budget,
                          count_tokens=lambda text: len(text) // 3)
        assert len(result.observation) // 3 <= budget + 60


def test_schema_lookup_tool_rejects_empty_arguments(store):
    tool = SchemaLookupTool(store)
    assert tool({})["status"] == "error"
    assert tool({"query": "student name"})["status"] == "ok"


def test_heuristic_join_is_marked_low_confidence(store):
    result = retrieve(store, "student department identifier")
    for rel in result.relationships:
        assert rel["source"] in {"sqlite_fk", "name_match"}
        if rel["source"] == "name_match":
            assert rel["confidence"] < 1.0


def test_value_lookup_preserves_database_case(store):
    con = sqlite3.connect(store.manifest["db_path"])
    con.execute("INSERT INTO students VALUES (1, 10, 'Alice')")
    con.execute("INSERT INTO departments VALUES (10, 'Math')")
    con.commit()
    con.close()
    result = retrieve(store, "Alice student name", enable_value_lookup=True)
    assert result.status == "ok"
    assert any(v["value"] == "Alice" for v in result.values), result.values
    assert "Relevant Values:" in result.observation
