"""CPU-only helpers for auditing hard SFT-pool questions and building schema bundles.

Nothing here calls a model, GPU, or external API. The helpers are deliberately
conservative: they extract textual evidence and SQLite metadata, but they do
not claim to resolve semantic ambiguity on their own.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "whose", "with", "all", "any", "each", "per", "than", "then",
    "there", "these", "those", "into", "over", "under", "between", "during",
    "select", "from", "where", "group", "order", "limit", "join", "inner",
    "left", "right", "outer", "on", "as", "distinct", "count", "sum", "avg",
    "min", "max", "cast", "real", "integer", "text", "null", "not", "like",
}

FORMAT_ONLY = re.compile(r"^(?:%[A-Za-z]|[-\s:/.])+$")


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def tokenize(text: str | None) -> list[str]:
    """Lowercase word tokens with snake_case and camelCase splitting."""
    if not text:
        return []
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str | None) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def normalize_text(text: str | None) -> str:
    return " ".join(content_tokens(text))


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def unquote_ident(name: str) -> str:
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in '`"':
        return name[1:-1]
    if len(name) >= 2 and name[0] == "[" and name[-1] == "]":
        return name[1:-1]
    return name


def sql_string_literals(sql: str | None) -> list[str]:
    """Return decoded single-quoted SQL string literals, in order."""
    if not sql:
        return []
    values = []
    for match in re.finditer(r"'((?:''|[^'])*)'", sql):
        value = match.group(1).replace("''", "'")
        if value:
            values.append(value)
    return values


def text_quoted_literals(text: str | None) -> list[str]:
    """Return quoted phrases from question/Evidence text.

    BIRD Evidence sometimes contains doubled quotes (``''Cystitis'``) because
    the source field is SQL-like. The negative lookbehind avoids treating an
    apostrophe inside a word (``aldermen's``) as an opening quote, and values
    shorter than three characters are ignored to avoid ``'O''Sullivan``-style
    fragments.
    """
    if not text:
        return []
    values = []
    patterns = [
        r"(?<![\w])'+(.*?)'+(?![\w])",
        r'"([^"]+)"',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).replace("''", "'").strip()
            if len(value) >= 3 and value not in values:
                values.append(value)
    return values


def is_format_only(value: str) -> bool:
    return bool(FORMAT_ONLY.match(value.strip()))


def sql_table_aliases(sql: str, table_names: set[str]) -> dict[str, str]:
    """Map aliases (and bare table names) in a Gold SQL to catalog table names."""
    aliases: dict[str, str] = {}
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+("
        r"`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w\-]*)"
        r"(?:\s+(?:AS\s+)?([A-Za-z_][\w]*))?",
        re.IGNORECASE,
    )
    stop_alias = {
        "ON", "WHERE", "GROUP", "ORDER", "LIMIT", "UNION", "EXCEPT",
        "INTERSECT", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "JOIN",
        "USING", "AS", "AND", "OR",
    }
    for match in pattern.finditer(sql):
        table = unquote_ident(match.group(1))
        if table.lower() not in table_names:
            continue
        alias = match.group(2)
        if alias and alias.upper() in stop_alias:
            alias = None
        aliases[table.lower()] = table
        if alias:
            aliases[alias.lower()] = table
    return aliases


def gold_join_edges(sql: str, table_names: set[str]) -> list[dict]:
    """Extract explicit equality joins from Gold SQL.

    This is intentionally shallow: it handles ``alias.a = alias.b`` and
    ``USING (col)``; it is used only to describe a training target, not to
    prove semantic correctness.
    """
    aliases = sql_table_aliases(sql, table_names)
    edges: list[dict] = []
    seen: set[tuple] = set()

    def add(left_table: str, left_col: str, right_table: str, right_col: str):
        if left_table.lower() == right_table.lower():
            return
        key = tuple(sorted([
            (left_table.lower(), left_col.lower()),
            (right_table.lower(), right_col.lower()),
        ]))
        if key in seen:
            return
        seen.add(key)
        edges.append({
            "left_table": left_table,
            "left_column": left_col,
            "right_table": right_table,
            "right_column": right_col,
            "source": "gold_sql_equality",
        })

    column_pattern = r"(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w]*)"
    for match in re.finditer(
            rf"([A-Za-z_][\w]*)\.{column_pattern}\s*=\s*"
            rf"([A-Za-z_][\w]*)\.{column_pattern}", sql):
        left_alias, left_col, right_alias, right_col = match.groups()
        left_col = unquote_ident(left_col)
        right_col = unquote_ident(right_col)
        left_table = aliases.get(left_alias.lower())
        right_table = aliases.get(right_alias.lower())
        if left_table and right_table:
            add(left_table, left_col, right_table, right_col)

    for match in re.finditer(r"\bUSING\s*\(\s*([A-Za-z_][\w]*)\s*\)", sql, re.I):
        col = match.group(1)
        tables = sorted(set(aliases.values()))
        if len(tables) >= 2:
            add(tables[0], col, tables[1], col)
    return edges


class SchemaCatalog:
    """Read-only SQLite schema, column type, and declared-FK metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(
            Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.conn.execute("PRAGMA query_only=ON")
        self.tables: "OrderedDict[str, dict]" = OrderedDict()
        self.table_lookup: dict[str, str] = {}
        for (name,) in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"):
            info = self.conn.execute(f"PRAGMA table_xinfo({quote_ident(name)})").fetchall()
            columns = []
            for row in info:
                hidden = bool(row[6]) if len(row) > 6 else False
                if hidden:
                    continue
                columns.append({
                    "name": row[1],
                    "type": (row[2] or "").upper(),
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                })
            fks = []
            for fk in self.conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})"):
                # (id, seq, table, from, to, on_update, on_delete, match)
                to_col = fk[4]
                if not to_col:
                    ref_info = self.conn.execute(
                        f"PRAGMA table_info({quote_ident(fk[2])})").fetchall()
                    pk = [r[1] for r in ref_info if r[5]]
                    to_col = pk[0] if pk else ""
                fks.append({
                    "from_table": name,
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": to_col,
                })
            self.tables[name] = {"name": name, "columns": columns, "fks": fks}
            self.table_lookup[name.lower()] = name
        self.fk_edges = []
        for table, info in self.tables.items():
            self.fk_edges.extend(info["fks"])
        self._name_match_edges = None

    def close(self):
        self.conn.close()

    def table(self, name: str) -> dict | None:
        actual = self.table_lookup.get(name.lower())
        return self.tables.get(actual) if actual else None

    def has_column(self, table: str, column: str) -> bool:
        info = self.table(table)
        if not info:
            return False
        return any(col["name"].lower() == column.lower() for col in info["columns"])

    def is_primary_key(self, table: str, column: str) -> bool:
        info = self.table(table)
        if not info:
            return False
        return any(col["pk"] and col["name"].lower() == column.lower()
                   for col in info["columns"])

    def column_tables(self, column: str) -> list[str]:
        wanted = column.lower()
        return [name for name, info in self.tables.items()
                if any(col["name"].lower() == wanted for col in info["columns"])]

    def fuzzy_columns(self, column: str) -> list[tuple[str, str]]:
        """Resolve Evidence names such as ``Order Quantity`` -> ``Quantity``.

        Only used after an exact column lookup fails; the caller decides
        whether a unique match is safe enough.
        """
        wanted = normalize_text(column)
        if not wanted:
            return []
        matches = []
        for table, info in self.tables.items():
            for col in info["columns"]:
                candidate = normalize_text(col["name"])
                if candidate and (wanted == candidate
                                  or wanted in candidate
                                  or candidate in wanted):
                    matches.append((table, col["name"]))
        return matches

    def name_match_edges(self) -> list[dict]:
        """Candidate joins from same-name columns; declared FKs are preferred."""
        if self._name_match_edges is not None:
            return self._name_match_edges
        by_column: dict[str, list[str]] = {}
        for table, info in self.tables.items():
            for col in info["columns"]:
                by_column.setdefault(col["name"].lower(), []).append(table)
        edges = []
        seen: set[tuple] = set()
        generic = {"id", "name", "type", "date", "year", "code", "value",
                   "description", "title", "state", "country", "city",
                   "address", "number", "count", "gender", "race", "party"}
        for column, tables in sorted(by_column.items()):
            if len(tables) < 2 or column in generic:
                continue
            # Same-name columns are plausible join keys when the name is
            # specific enough. ``unitid`` is a real BIRD join key; ``year`` is
            # not, which is why the generic list above is excluded.
            if not (column.endswith("_id") or column.endswith("id")
                    or len(set(tables)) <= 3):
                continue
            for index, left in enumerate(sorted(set(tables))):
                for right in sorted(set(tables))[index + 1:]:
                    key = tuple(sorted([(left.lower(), column), (right.lower(), column)]))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "left_table": left,
                        "left_column": column,
                        "right_table": right,
                        "right_column": column,
                        "source": "name_match",
                    })

        def singular(name: str) -> str:
            name = name.lower()
            if name.endswith("ies") and len(name) > 3:
                return name[:-3] + "y"
            if name.endswith("ses") and len(name) > 3:
                return name[:-2]
            if name.endswith("s") and len(name) > 2:
                return name[:-1]
            return name

        table_norms = {name: normalize_text(name) for name in self.tables}
        target_pks = {}
        for target, info in self.tables.items():
            pk_columns = [col["name"] for col in info["columns"] if col["pk"]]
            target_pks[target] = pk_columns[0] if pk_columns else next(
                (col["name"] for col in info["columns"]
                 if col["name"].lower() == "id"), None)
        for table, info in self.tables.items():
            for col in info["columns"]:
                name = col["name"].lower()
                if not name.endswith("_id"):
                    continue
                base = normalize_text(name[:-3])
                if not base:
                    continue
                for target, target_norm in table_norms.items():
                    if target == table or not target_norm:
                        continue
                    if base not in {target_norm, singular(target_norm)}:
                        continue
                    target_pk = target_pks.get(target)
                    if not target_pk:
                        continue
                    key = tuple(sorted([(table.lower(), name.lower()),
                                        (target.lower(), target_pk.lower())]))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "left_table": table,
                        "left_column": col["name"],
                        "right_table": target,
                        "right_column": target_pk,
                        "source": "fk_name_prefix",
                    })
        self._name_match_edges = edges
        return edges

    def fk_path(self, source: str, target: str, max_hops: int = 4) -> list[dict]:
        source = self.table_lookup.get(source.lower())
        target = self.table_lookup.get(target.lower())
        if not source or not target:
            return []
        if source == target:
            return []
        adjacency: dict[str, list[tuple[str, dict]]] = {}
        for edge in self.fk_edges:
            adjacency.setdefault(edge["from_table"], []).append((edge["to_table"], edge))
            adjacency.setdefault(edge["to_table"], []).append((edge["from_table"], edge))
        queue = [(source, [])]
        visited = {source}
        while queue:
            node, path = queue.pop(0)
            if len(path) >= max_hops:
                continue
            for neighbor, edge in adjacency.get(node, []):
                if neighbor in visited:
                    continue
                next_path = path + [edge]
                if neighbor == target:
                    return next_path
                visited.add(neighbor)
                queue.append((neighbor, next_path))
        return []


def table_document(catalog: SchemaCatalog, table: str) -> tuple[set[str], set[str]]:
    info = catalog.tables[table]
    name_tokens = set(tokenize(table))
    column_tokens = set()
    for col in info["columns"]:
        column_tokens.update(tokenize(col["name"]))
    return name_tokens, column_tokens


def rank_tables(question: str, evidence: str, catalog: SchemaCatalog,
                top_k: int = 6, max_tables: int = 9) -> tuple[list[str], dict[str, float]]:
    """Rank catalog tables by lexical overlap, with IDF weighting."""
    text = f"{question or ''} {evidence or ''}"
    q_tokens = set(content_tokens(text))
    docs = {table: table_document(catalog, table) for table in catalog.tables}
    df: Counter = Counter()
    for name_tokens, column_tokens in docs.values():
        for token in name_tokens | column_tokens:
            df[token] += 1
    total = max(len(docs), 1)
    idf = {token: math.log((total + 1) / (count + 1)) + 1.0
           for token, count in df.items()}
    normalized_text = normalize_text(text)
    refs = evidence_table_refs(evidence, catalog)
    scores: dict[str, float] = {}
    for table, (name_tokens, column_tokens) in docs.items():
        score = 0.0
        if normalize_text(table) and normalize_text(table) in normalized_text:
            score += 8.0
        if table.lower() in (question or "").lower():
            score += 3.0
        for token in name_tokens & q_tokens:
            score += 3.0 * idf.get(token, 1.0)
        for token in column_tokens & q_tokens:
            score += 1.0 * idf.get(token, 1.0)
        if table in refs:
            score += 5.0
        scores[table] = round(score, 4)
    ordered = sorted(scores, key=lambda name: (-scores[name], name.lower()))
    selected = [name for name in ordered if scores[name] > 0][:top_k]
    if not selected:
        # No lexical match: keep a small fallback rather than dumping the
        # whole catalog. The caller can inspect table_scores.
        selected = ordered[:min(3, len(ordered))]
    # Add one-hop FK neighbors of the lexically selected tables. Do not expand
    # transitively: one hub table would otherwise pull in most of the database.
    selected_set = set(selected)
    seeds = set(selected)
    for edge in catalog.fk_edges:
        left, right = edge["from_table"], edge["to_table"]
        if left in seeds and right not in selected_set:
            selected.append(right)
            selected_set.add(right)
        elif right in seeds and left not in selected_set:
            selected.append(left)
            selected_set.add(left)
        if len(selected) >= max_tables:
            break
    return selected, scores


def select_columns(catalog: SchemaCatalog, table: str, question: str,
                   evidence: str, max_columns: int = 16,
                   peer_tables: set[str] | None = None) -> list[str]:
    info = catalog.table(table)
    if not info:
        return []
    q_tokens = set(content_tokens(f"{question or ''} {evidence or ''}"))
    refs = evidence_column_refs(evidence, catalog)
    ref_columns = {col.lower() for tbl, col in refs if tbl == table}
    scored = []
    for col in info["columns"]:
        col_tokens = set(tokenize(col["name"]))
        score = 0
        if col["name"].lower() in ref_columns:
            score += 10
        if normalize_text(col["name"]) and normalize_text(col["name"]) in normalize_text(
                f"{question or ''} {evidence or ''}"):
            score += 5
        score += len(col_tokens & q_tokens)
        if score:
            scored.append((score, col["name"]))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    must = {col["name"] for col in info["columns"] if col["pk"]}
    peer_tables = set(peer_tables) if peer_tables is not None else None
    for edge in catalog.fk_edges:
        if edge["from_table"] == table and (
                peer_tables is None or edge["to_table"] in peer_tables):
            must.add(edge["from_column"])
        if edge["to_table"] == table and (
                peer_tables is None or edge["from_table"] in peer_tables):
            must.add(edge["to_column"])
    for edge in catalog.name_match_edges():
        if edge["left_table"] == table and (
                peer_tables is None or edge["right_table"] in peer_tables):
            if edge.get("source") == "fk_name_prefix" or catalog.is_primary_key(
                    edge["left_table"], edge["left_column"]):
                must.add(edge["left_column"])
        if edge["right_table"] == table and (
                peer_tables is None or edge["left_table"] in peer_tables):
            if edge.get("source") == "fk_name_prefix" or catalog.is_primary_key(
                    edge["right_table"], edge["right_column"]):
                must.add(edge["right_column"])
    # Put the most question-relevant columns first, then join/primary keys,
    # then the remaining lexical matches. This keeps the cap from dropping the
    # SELECT/WHERE column that the question actually names.
    ordered = []
    for _, name in scored[:8]:
        if name not in ordered:
            ordered.append(name)
    for name in sorted(must):
        if name not in ordered:
            ordered.append(name)
    for _, name in scored[8:]:
        if name not in ordered:
            ordered.append(name)
    if len(info["columns"]) <= max_columns:
        ordered = [col["name"] for col in info["columns"]]
    return ordered[:max_columns]


def evidence_table_refs(evidence: str | None, catalog: SchemaCatalog) -> set[str]:
    refs = set()
    if not evidence:
        return refs
    for table, _column in evidence_column_refs(evidence, catalog):
        refs.add(table)
    return refs


def evidence_column_refs(evidence: str | None, catalog: SchemaCatalog) -> list[tuple[str, str]]:
    """Resolve ``table.column`` references in Evidence against the catalog."""
    refs: list[tuple[str, str]] = []
    if not evidence:
        return refs
    for match in re.finditer(
            r"(?<![\w.])([A-Za-z_][\w\-]*)\.([A-Za-z_][\w\-]*)(?![\w.])", evidence):
        table_raw, column = match.groups()
        if re.fullmatch(r"[Tt]\d+", table_raw):
            continue
        info = catalog.table(table_raw)
        if info and catalog.has_column(info["name"], column):
            refs.append((info["name"], column))
    return refs


def extract_evidence_filters(evidence: str | None) -> list[dict]:
    """Parse simple ``column = value`` / ``column > number`` Evidence hints."""
    filters = []
    if not evidence:
        return filters
    pattern = re.compile(
        r"([A-Za-z_][\w\-]*(?:\.[A-Za-z_][\w\-]*)?)\s*(>=|<=|=|>|<)\s*"
        r"('{1,2}[^']+'{1,2}|\"[^\"]+\"|\d+(?:\.\d+)?)",
    )
    for match in pattern.finditer(evidence):
        raw_column, op, raw_value = match.groups()
        value = raw_value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')):
            value = value.strip("'\"")
        value = value.replace("''", "'").strip()
        if not value:
            continue
        table_hint = None
        column = raw_column
        if "." in raw_column:
            table_hint, column = raw_column.split(".", 1)
        filters.append({
            "table_hint": table_hint,
            "column": column,
            "op": op,
            "value": value,
        })
    return filters


def resolve_filter_column(catalog: SchemaCatalog, item: dict) -> tuple[str | None, str | None]:
    """Return (table, actual_column) for an Evidence filter, if resolvable."""
    table_hint = item.get("table_hint")
    column = item.get("column", "")
    if table_hint:
        info = catalog.table(table_hint)
        if info:
            for col in info["columns"]:
                if col["name"].lower() == column.lower():
                    return info["name"], col["name"]
        return info["name"] if info else None, None
    candidates = catalog.column_tables(column)
    if len(candidates) == 1:
        info = catalog.tables[candidates[0]]
        for col in info["columns"]:
            if col["name"].lower() == column.lower():
                return info["name"], col["name"]
    if not candidates:
        fuzzy = catalog.fuzzy_columns(column)
        if len(fuzzy) == 1:
            return fuzzy[0]
    return None, None


def gold_requirements(db_path: str | Path, sql: str) -> dict:
    """Call the project's SQLite authorizer-based dependency resolver."""
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eval.sql_dependencies import sqlite_requirements
    return sqlite_requirements(db_path, sql)


def dialect_table_names(requirements: dict, catalog: SchemaCatalog) -> list[str]:
    return [catalog.table_lookup.get(name.lower(), name)
            for name in requirements.get("tables", [])]


def dialect_columns(requirements: dict, catalog: SchemaCatalog) -> list[dict]:
    columns = []
    for item in requirements.get("columns", []):
        if "." not in item:
            continue
        table_raw, column_raw = item.split(".", 1)
        table = catalog.table_lookup.get(table_raw.lower(), table_raw)
        info = catalog.table(table)
        actual = column_raw
        if info:
            for col in info["columns"]:
                if col["name"].lower() == column_raw.lower():
                    actual = col["name"]
                    break
        columns.append({"table": table, "column": actual})
    return columns


def execute_gold(db_path: str | Path, sql: str, timeout_s: float = 4.0) -> dict:
    result = {
        "ok": False,
        "rows": None,
        "empty": None,
        "error": None,
        "timeout": False,
        "elapsed_s": None,
    }
    started = time.monotonic()
    try:
        con = sqlite3.connect(
            Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        con.execute("PRAGMA query_only=ON")
        deadline = started + timeout_s
        con.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 10000)
        try:
            rows = con.execute(sql).fetchmany(1001)
        finally:
            con.set_progress_handler(None, 0)
            con.close()
        result.update({
            "ok": True,
            "rows": len(rows),
            "empty": len(rows) == 0,
            "elapsed_s": round(time.monotonic() - started, 3),
        })
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["error"] = message
        result["timeout"] = "interrupted" in message.lower() or "timeout" in message.lower()
        result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def compact_schema_lines(catalog: SchemaCatalog, tables: list[str],
                         selected_columns: dict[str, list[str]] | None = None) -> list[str]:
    lines = []
    for table in tables:
        info = catalog.table(table)
        if not info:
            continue
        keep = selected_columns.get(table) if selected_columns else None
        cols = []
        for col in info["columns"]:
            if keep is not None and col["name"] not in keep:
                continue
            marker = " PK" if col["pk"] else ""
            cols.append(f"{col['name']} {col['type']}{marker}".strip())
        lines.append(f"TABLE {info['name']} ({', '.join(cols)})")
    return lines
