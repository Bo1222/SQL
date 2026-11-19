"""schema_lookup tool: one query in, one unified Observation out."""
import time

from retrieval.e1_1 import RETRIEVAL_VERSION as E1_1_VERSION
from retrieval.e1_1 import retrieve_e1_1
from retrieval.e1_2 import retrieve_e1_2
from retrieval.retriever import retrieve


class SchemaLookupTool:
    """Wraps a retrieval backend so the agent sees exactly one Observation."""

    def __init__(self, store, token_budget=1536, count_tokens=None, enable_value_lookup=False,
                 top_k_columns=10, max_tables=6, hops=2, retrieval_version="E1"):
        self.store = store
        self.token_budget = token_budget
        self.count_tokens = count_tokens
        self.enable_value_lookup = enable_value_lookup
        self.top_k_columns = top_k_columns
        self.max_tables = max_tables
        self.hops = hops
        self.retrieval_version = retrieval_version

    def __call__(self, arguments):
        query = arguments.get("query") if isinstance(arguments, dict) else None
        table = arguments.get("table") if isinstance(arguments, dict) else None
        if not query and not table:
            return {"status": "error", "error_type": "schema_error",
                    "message": 'schema_lookup requires {"query": "<what you need>"}'}
        started = time.monotonic()
        table_hint = table if not query else None
        kwargs = dict(token_budget=self.token_budget, count_tokens=self.count_tokens,
                      enable_value_lookup=self.enable_value_lookup,
                      top_k_columns=self.top_k_columns, max_tables=self.max_tables,
                      hops=self.hops, table_hint=table_hint)
        if self.retrieval_version == "E1.2":
            # Explicit column inspection must honor the table even when a
            # query is supplied; never accept and silently ignore the column.
            kwargs["table_hint"] = table
            result = retrieve_e1_2(self.store, query, column_hint=arguments.get("column"), **kwargs)
        elif self.retrieval_version == E1_1_VERSION:
            # Keep query=None for a table-only lookup; retrieve_e1_1 must see the
            # hint rather than treating the table name as a BM25 query.
            result = retrieve_e1_1(self.store, query, **kwargs)
        else:
            result = retrieve(self.store, query or table, **kwargs)
        if result.status == "error":
            return {"status": "error", "error_type": "schema_error", "message": result.observation}
        metadata = result.to_dict()["retrieval_metadata"]
        metadata.setdefault("retrieval_version", self.retrieval_version)
        return {"status": "ok", "observation": result.observation,
                "retrieval_metadata": metadata,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
