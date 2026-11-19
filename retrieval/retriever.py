"""Unified retrieval backend for `schema_lookup`.

One query -> one Observation containing validated tables/columns, descriptions,
PK/FK join relationships and optional value hints.
"""
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.catalog_store import words
from retrieval.observation_formatter import format_observation, default_count_tokens

K1, B = 1.2, 0.75


@dataclass
class RetrievalResult:
    status: str = "ok"
    observation: str = ""
    tables: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    values: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {"status": self.status, "observation": self.observation,
                "retrieval_metadata": {"matched_tables": self.tables, "matched_columns": self.columns,
                                       "relationships": self.relationships, "values": self.values,
                                       **self.diagnostics}}


def _bm25(store, tokens):
    index = store.index
    n, avgdl, df = index["n_docs"], index["avgdl"], index["df"]
    scores = []
    for i, doc in enumerate(index["docs"]):
        tokens_doc = doc["tokens"]
        if not tokens_doc:
            continue
        tf = {}
        for t in tokens_doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(tokens_doc)
        score = 0.0
        for q in tokens:
            f = tf.get(q, 0)
            if not f:
                continue
            idf = __import__("math").log(1 + (n - df.get(q, 0) + .5) / (df.get(q, 0) + .5))
            score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / max(avgdl, 1e-6)))
        if score > 0:
            scores.append((score, i))
    scores.sort(key=lambda x: -x[0])
    return scores


def _expand_relationships(store, selected_tables, hops=2, max_extra_tables=3):
    """BFS over the verified join graph; return (tables, relationships)."""
    seen = list(selected_tables)
    queue = [(t, 0) for t in selected_tables]
    edges = []
    edge_keys = set()
    while queue:
        table, depth = queue.pop(0)
        if depth >= hops:
            continue
        for rel in store.adjacency.get(table, []):
            key = (rel["from_table"], rel["from_column"], rel["to_table"], rel["to_column"])
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append({"from_table": rel["from_table"], "from_column": rel["from_column"],
                              "to_table": rel["to_table"], "to_column": rel["to_column"],
                              "source": rel.get("source", "sqlite_fk"),
                              "confidence": rel.get("confidence", 1.0)})
            other = rel["to_table"]
            if other not in seen:
                if len(seen) >= len(selected_tables) + max_extra_tables:
                    continue
                seen.append(other)
                queue.append((other, depth + 1))
    keep = set(seen)
    edges = [e for e in edges if e["from_table"] in keep and e["to_table"] in keep]
    return seen, edges


def _value_lookup(store, tokens, columns, timeout=2.0, limit=5):
    """Bounded value retrieval inside the selected columns only.

    BIRD evidence often uses a different case/number than the database value
    (``subscriber`` vs ``Subscriber``, ``rainy`` vs ``Rain``).  Try a small set
    of singular/stem variants so the model sees the database's actual value.
    """
    db_path = store.manifest.get("db_path") or store.schema.get("db_path")
    if not db_path or not Path(db_path).is_file():
        return [], {"value_lookup_error": "database missing"}
    variants = []
    for token in tokens[:4]:
        candidates = [token]
        if len(token) > 3 and token.endswith("s"):
            candidates.append(token[:-1])
        if len(token) > 3 and token.endswith("y"):
            candidates.append(token[:-1])
        for candidate in candidates:
            if candidate and candidate not in variants:
                variants.append(candidate)
    variants = variants[:6]
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=timeout)
    hits, errors = [], []
    try:
        con.execute("PRAGMA query_only = ON")
        for table, column in columns[:4]:
            for token in variants:
                sql = (f'SELECT DISTINCT "{column}" FROM "{table}" '
                       f'WHERE CAST("{column}" AS TEXT) LIKE ? LIMIT {int(limit)}')
                started = time.monotonic()
                try:
                    rows = con.execute(sql, (f"%{token}%",)).fetchall()
                    if time.monotonic() - started > timeout:
                        errors.append({"table": table, "column": column, "error": "timeout"})
                        continue
                    for (value,) in rows:
                        hits.append({"table": table, "column": column, "value": str(value)[:80]})
                except sqlite3.Error as exc:
                    errors.append({"table": table, "column": column, "error": str(exc)[:120]})
    finally:
        con.close()
    dedup = list({(h["table"], h["column"], h["value"]): h for h in hits}.values())[:8]
    return dedup, ({"value_lookup_errors": errors[:5]} if errors else {})


def _heuristic_joins(store, tables, limit=4):
    """Low-confidence join candidates from identical column names (never labelled FK)."""
    if len(tables) < 2:
        return []
    columns = {t: {c["column"].lower() for c in store.catalog if c["table"] == t} for t in tables}
    edges, ordered = [], sorted(tables)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            for column in sorted(columns[left] & columns[right]):
                if column == "id" or column.endswith("_id") or column in {"zip_code", "code"}:
                    edges.append({"from_table": left, "from_column": column,
                                  "to_table": right, "to_column": column,
                                  "source": "name_match", "confidence": 0.5})
                if len(edges) >= limit:
                    return edges
    return edges


def retrieve(store, query, token_budget=1536, count_tokens=None, enable_value_lookup=False,
             top_k_columns=10, max_tables=6, hops=2, table_hint=None):
    """Run the unified retrieval pipeline and return one Observation."""
    started = time.monotonic()
    count_tokens = count_tokens or default_count_tokens
    tokens = words(query or "")
    diagnostics = {"query": query, "query_tokens": tokens, "fallback_used": False,
                   "value_lookup_enabled": bool(enable_value_lookup), "store_version": store.manifest.get("store_version")}

    if table_hint:
        table = next((t for t in store.table_names if t.lower() == str(table_hint).lower()), None)
        if table is None:
            return RetrievalResult(status="error", observation=f"Unknown table: {table_hint}",
                                   diagnostics={**diagnostics, "error": "unknown_table",
                                                "latency_ms": round((time.monotonic() - started) * 1000, 2)})
        scored = [(1.0, (table, c["column"])) for c in store.catalog if c["table"] == table]
    else:
        pairs = _bm25(store, tokens)
        scored = [(score, (store.index["docs"][i]["table"], store.index["docs"][i]["column"])) for score, i in pairs]

    if not scored:
        # fallback: raw substring match on real schema names
        diagnostics["fallback_used"] = True
        lowered = [t.lower() for t in tokens]
        scored = []
        for record in store.catalog:
            name = f"{record['table']}.{record['column']}".lower()
            if any(t in name for t in lowered):
                scored.append((0.05, (record["table"], record["column"])))
        if not scored:
            return RetrievalResult(status="empty",
                                   observation="No relevant database context found; refine the query.",
                                   diagnostics={**diagnostics, "empty": True,
                                                "latency_ms": round((time.monotonic() - started) * 1000, 2)})

    top = scored[:top_k_columns]
    table_scores = {}
    for score, (table, _) in top:
        table_scores[table] = table_scores.get(table, 0.0) + score
    ordered_tables = sorted(table_scores, key=lambda t: -table_scores[t])
    selected_tables = ordered_tables[:max_tables]
    selected_columns = []
    for score, (table, column) in top:
        if table in selected_tables:
            selected_columns.append({"table": table, "column": column, "score": round(score, 4)})

    selected_tables, relationships = _expand_relationships(store, selected_tables, hops=hops)

    declared = {(r["from_table"], r["from_column"], r["to_table"], r["to_column"]) for r in relationships}
    for edge in _heuristic_joins(store, selected_tables):
        key = (edge["from_table"], edge["from_column"], edge["to_table"], edge["to_column"])
        if key not in declared and (key[2], key[3], key[0], key[1]) not in declared:
            relationships.append(edge)

    # guarantee join keys of kept relationships are present
    keyed = {(c["table"], c["column"]) for c in selected_columns}
    for rel in relationships:
        for table, column in ((rel["from_table"], rel["from_column"]), (rel["to_table"], rel["to_column"])):
            if (table, column) not in keyed:
                selected_columns.append({"table": table, "column": column, "score": 0.0, "join_key": True})
                keyed.add((table, column))

    values, value_diag = ([], {})
    if enable_value_lookup:
        values, value_diag = _value_lookup(store, tokens,
                                           [(c["table"], c["column"]) for c in selected_columns
                                            if store.column_types.get((c["table"], c["column"]), "").upper() in ("TEXT", "", "VARCHAR")])

    observation = format_observation(store, selected_tables, selected_columns, relationships, values,
                                     token_budget=token_budget, count_tokens=count_tokens)
    diagnostics.update(value_diag)
    diagnostics.update({
        "candidate_columns": len(scored),
        "selected_tables": len(selected_tables),
        "selected_columns": len(selected_columns),
        "relationship_count": len(relationships),
        "observation_tokens": count_tokens(observation),
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
    })
    return RetrievalResult(status="ok", observation=observation, tables=selected_tables,
                           columns=[{k: v for k, v in c.items() if k != "score"} for c in selected_columns],
                           relationships=relationships, values=values, diagnostics=diagnostics)



