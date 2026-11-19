"""Opt-in E1.2: honor exact table/column inspection requests.

E1/E1.1 remain reproducible. Query-only requests retain E1.1 behavior; an
explicit table request is no longer silently capped at the first ten columns,
and an explicit column cannot be silently ignored. This does not use gold.
"""
from retrieval.e1_1 import retrieve_e1_1
from retrieval.observation_formatter import format_observation
from retrieval.retriever import RetrievalResult

RETRIEVAL_VERSION = "E1.2"


def retrieve_e1_2(store, query=None, *, table_hint=None, column_hint=None,
                  token_budget=1536, count_tokens=None, **kwargs):
    diagnostics = {"retrieval_version": RETRIEVAL_VERSION,
                   "table_hint": table_hint, "column_hint": column_hint}
    if not table_hint:
        if column_hint:
            return RetrievalResult(status="error", observation="A column lookup requires a table name.",
                                   diagnostics=diagnostics)
        result = retrieve_e1_1(store, query, token_budget=token_budget,
                               count_tokens=count_tokens, **kwargs)
        result.diagnostics.update(diagnostics)
        return result
    table = next((t for t in store.table_names if t.lower() == str(table_hint).lower()), None)
    if table is None:
        return RetrievalResult(status="error", observation=f"Unknown table: {table_hint}",
                               diagnostics=diagnostics)
    records = [c for c in store.catalog if c["table"] == table]
    if column_hint:
        records = [c for c in records if c["column"].lower() == str(column_hint).lower()]
        if not records:
            return RetrievalResult(status="error", observation=f"Unknown column: {table}.{column_hint}",
                                   diagnostics=diagnostics)
    selected = [{"table": table, "column": c["column"], "score": 1.0} for c in records]
    # Do not expand unrelated tables into an exact inspection request. Users
    # can perform query-based relationship retrieval separately.
    observation = format_observation(store, [table], selected, [], [],
                                     token_budget=token_budget, count_tokens=count_tokens)
    diagnostics.update({"inspection": True, "selected_columns": len(selected),
                        "available_table_columns": sum(c["table"] == table for c in store.catalog),
                        "empty": False})
    return RetrievalResult(status="ok", observation=observation, tables=[table],
                           columns=[{"table": c["table"], "column": c["column"]} for c in selected],
                           diagnostics=diagnostics)
