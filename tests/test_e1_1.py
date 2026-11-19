import pytest

from retrieval.e1_1 import (
    RETRIEVAL_VERSION,
    _table_fallback,
    normalize_query,
    retrieve_e1_1,
)
from tools.schema_lookup import SchemaLookupTool


class FakeStore:
    def __init__(self):
        self.table_names = ["city", "country"]
        self.catalog = [
            {"table": "city", "column": "name", "description": "city name"},
            {"table": "city", "column": "country_id", "description": "country id"},
            {"table": "country", "column": "name", "description": "country name"},
        ]
        self.catalog_by_column = {(c["table"], c["column"]): c for c in self.catalog}
        self.column_types = {("city", "name"): "TEXT", ("city", "country_id"): "INTEGER",
                             ("country", "name"): "TEXT"}
        docs = [
            {"table": c["table"], "column": c["column"],
             "text": f"{c['table']} {c['column']} {c['description']}",
             "tokens": (c["table"] + " " + c["column"] + " " + c["description"]).split()}
            for c in self.catalog
        ]
        df = {}
        for doc in docs:
            for token in set(doc["tokens"]):
                df[token] = df.get(token, 0) + 1
        self.index = {"docs": docs, "df": df, "n_docs": len(docs),
                      "avgdl": sum(len(d["tokens"]) for d in docs) / len(docs)}
        self.adjacency = {"city": [], "country": []}
        self.manifest = {"store_version": "v1", "db_path": None}
        self.schema = {"tables": [{"name": t} for t in self.table_names]}


def test_normalize_query_stems_plurals():
    assert "city" in normalize_query("List the cities").split()
    assert "country" in normalize_query("countries").split()
    assert "brewery" in normalize_query("breweries").split()


def test_e1_1_retries_normalized_query():
    store = FakeStore()
    result = retrieve_e1_1(store, "cities", token_budget=512,
                           count_tokens=lambda text: len(text.split()))
    assert result.status == "ok"
    assert result.diagnostics["retrieval_version"] == RETRIEVAL_VERSION
    assert result.diagnostics["e1_1_used"] is True
    assert result.diagnostics["fallback_reason"] == "normalized_query"
    assert "city" in result.tables


def test_e1_1_table_fallback_never_empty():
    store = FakeStore()
    result = retrieve_e1_1(store, "zzzz", token_budget=512,
                           count_tokens=lambda text: len(text.split()))
    assert result.status == "ok"
    assert result.diagnostics["e1_1_used"] is True
    assert result.diagnostics["fallback_reason"] == "catalog_tables"
    assert result.tables == ["city", "country"]
    assert "Tables:" in result.observation
    assert "city" in result.observation


def test_table_fallback_prefers_overlap():
    store = FakeStore()
    selected, score = _table_fallback(store, "country name", max_tables=2,
                                      token_budget=512,
                                      count_tokens=lambda text: len(text.split()))
    assert selected[0] == "country"
    assert score > 0


def test_table_only_lookup_uses_hint_not_bm25():
    store = FakeStore()
    tool = SchemaLookupTool(store, token_budget=512, retrieval_version="E1.1")
    result = tool({"table": "country"})
    assert result["status"] == "ok"
    assert result["retrieval_metadata"]["matched_tables"] == ["country"]


def test_e1_2_exact_column_after_first_ten_is_discoverable():
    store = FakeStore()
    for index in range(20):
        item = {"table": "city", "column": f"extra_{index}", "description": ""}
        store.catalog.append(item)
        store.catalog_by_column[("city", item["column"])] = item
        store.column_types[("city", item["column"])] = "TEXT"
    old = SchemaLookupTool(store, token_budget=4096, retrieval_version="E1.1")
    new = SchemaLookupTool(store, token_budget=4096, retrieval_version="E1.2")
    arguments = {"table": "city", "column": "extra_19"}
    assert "extra_19" not in old(arguments)["observation"]
    result = new(arguments)
    assert result["status"] == "ok"
    assert "extra_19" in result["observation"]
    assert result["retrieval_metadata"]["matched_columns"] == [{"table": "city", "column": "extra_19"}]
    assert "extra_19" in new({"table": "city"})["observation"]


def test_e1_2_rejects_unknown_column_instead_of_silently_ignoring_it():
    tool = SchemaLookupTool(FakeStore(), retrieval_version="E1.2")
    result = tool({"table": "city", "column": "does_not_exist"})
    assert result["status"] == "error"
    assert "Unknown column" in result["message"]
