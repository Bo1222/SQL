"""E1.1 query handling: normalization + deterministic table fallback.

E1.1 is intentionally a thin, versioned wrapper around the frozen E1 retriever.
It does **not** change the BM25 backend, the catalog store or the observation
formatter.  It only acts when E1 returns ``status == "empty"``:

1. retry with simple singular/stem variants (``cities`` -> ``city``);
2. if still empty, return a table-level fallback observation with the
   best-matching table names, so the model can issue a table-specific lookup
   instead of receiving ``No relevant database context found``.

Every result records ``retrieval_version`` and a normalizer fingerprint.
"""
from __future__ import annotations

import hashlib
import re

from retrieval.catalog_store import words
from retrieval.observation_formatter import format_observation
from retrieval.retriever import RetrievalResult, retrieve

RETRIEVAL_VERSION = "E1.1"
NORMALIZER_VERSION = "e1.1-normalizer-v1"

_STEM_RULES = (
    ("ies", "y"),
    ("ves", "f"),
    ("ses", "s"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
)


def normalizer_fingerprint() -> str:
    payload = "|".join([NORMALIZER_VERSION, *("%s>%s" % pair for pair in _STEM_RULES)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


NORMALIZER_SHA256 = normalizer_fingerprint()


def _variants(token: str) -> list[str]:
    variants = [token]
    for suffix, replacement in _STEM_RULES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            variants.append(token[: -len(suffix)] + replacement)
            break
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.append(token[:-1])
    return list(dict.fromkeys(variants))


def normalize_query(query: str) -> str:
    """Return a stem-expanded query while keeping the original word order."""
    tokens = words(query or "")
    expanded = []
    for token in tokens:
        expanded.extend(_variants(token))
    return " ".join(dict.fromkeys(expanded))


def _table_fallback(store, query: str, *, max_tables: int, token_budget: int,
                    count_tokens):
    """Score tables by token overlap with the question and return a table list."""
    query_tokens = set()
    for token in words(query or ""):
        query_tokens.update(_variants(token))
    scores = {}
    for table in store.table_names:
        table_tokens = set(words(table))
        score = len(query_tokens & table_tokens)
        for record in store.catalog:
            if record.get("table") != table:
                continue
            column_tokens = set(words(record.get("column", "")))
            score += 0.25 * len(query_tokens & column_tokens)
        scores[table] = score
    ordered = sorted(scores, key=lambda t: (-scores[t], t))
    if not ordered:
        return [], 0.0
    best = scores[ordered[0]]
    selected = [t for t in ordered if scores[t] > 0][:max_tables]
    if not selected:
        # Deterministic catalog fallback: expose table names only.  The model
        # can then issue a table-specific schema_lookup.
        selected = ordered[:max_tables]
    return selected, float(best)


def retrieve_e1_1(store, query, token_budget=1536, count_tokens=None,
                  enable_value_lookup=False, top_k_columns=10, max_tables=6,
                  hops=2, table_hint=None):
    """Run E1 first; only on empty, apply normalization and table fallback."""
    hint = table_hint if not query else None
    result = retrieve(store, query, token_budget=token_budget,
                      count_tokens=count_tokens,
                      enable_value_lookup=enable_value_lookup,
                      top_k_columns=top_k_columns, max_tables=max_tables,
                      hops=hops, table_hint=hint)

    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "retrieval_version": RETRIEVAL_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "normalizer_fingerprint": NORMALIZER_SHA256,
        "e1_1_used": False,
        "e1_empty": result.status == "empty",
    })
    if result.status != "empty":
        result.diagnostics = diagnostics
        return result

    # Step 1: stem-expanded retry.
    normalized = normalize_query(query)
    if normalized and normalized != (query or "").strip().lower():
        retried = retrieve(store, normalized, token_budget=token_budget,
                           count_tokens=count_tokens,
                           enable_value_lookup=enable_value_lookup,
                           top_k_columns=top_k_columns, max_tables=max_tables,
                           hops=hops, table_hint=None)
        if retried.status != "empty":
            retried.diagnostics = {
                **diagnostics,
                **retried.diagnostics,
                "e1_1_used": True,
                "fallback_reason": "normalized_query",
                "normalized_query": normalized,
            }
            return retried

    # Step 2: table-level fallback.
    selected, best_score = _table_fallback(
        store, query, max_tables=max_tables, token_budget=token_budget,
        count_tokens=count_tokens)
    if not selected:
        result.diagnostics = {**diagnostics, "e1_1_used": True,
                              "fallback_reason": "no_tables"}
        return result

    observation = format_observation(
        store, selected, [], [], [],
        token_budget=token_budget,
        count_tokens=count_tokens or (lambda text: max(1, int(len(text) / 3.2))),
    )
    fallback = RetrievalResult(
        status="ok",
        observation=observation,
        tables=selected,
        columns=[],
        relationships=[],
        values=[],
        diagnostics={
            **diagnostics,
            "e1_1_used": True,
            "fallback_reason": "catalog_tables",
            "fallback_score": best_score,
            "matched_tables": selected,
            "selected_tables": len(selected),
            "selected_columns": 0,
            "relationship_count": 0,
            "empty": False,
            "observation_tokens": (count_tokens or (lambda text: max(1, int(len(text) / 3.2))))(observation),
            "latency_ms": diagnostics.get("latency_ms", 0),
        },
    )
    return fallback
