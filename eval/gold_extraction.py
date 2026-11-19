"""Extract gold schema dependencies for evaluation, never for policy inputs.

Physical table/column dependencies use SQLite authorization at prepare time.
The legacy join-name scan remains separately labelled as a heuristic.
"""
import re
import sqlite3
from pathlib import Path

STRING_RE = re.compile(r"'(?:[^']|'')*'")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
PLAN_TABLE_RE = re.compile(r"(?:SCAN|SEARCH)\s+(?:TABLE\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.I)
FROM_JOIN_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)"
                          r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?", re.I)
EQ_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                   r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")

SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit", "offset", "join", "inner",
    "left", "right", "full", "outer", "cross", "on", "as", "and", "or", "not", "in", "is", "null",
    "like", "between", "case", "when", "then", "else", "end", "distinct", "count", "sum", "avg",
    "min", "max", "cast", "integer", "real", "text", "date", "datetime", "strftime", "substr",
    "lower", "upper", "replace", "round", "abs", "coalesce", "ifnull", "union", "all", "with",
    "asc", "desc", "using", "natural", "except", "intersect", "exists", "over", "partition",
    "then", "when", "not", "null", "true", "false", "interval", "year", "month", "day",
}


def _connect(db_path):
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def schema_columns(db_path):
    """{(table_lower, column_lower)} plus {column_lower: set(tables)}."""
    pairs, by_column, tables = set(), {}, []
    con = _connect(db_path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            quoted = table.replace('"', '""')
            for row in con.execute(f'PRAGMA table_info("{quoted}")'):
                column = row[1]
                pairs.add((table.lower(), column.lower()))
                by_column.setdefault(column.lower(), set()).add(table.lower())
    finally:
        con.close()
    return pairs, by_column, {t.lower() for t in tables}


def alias_map(sql):
    mapping = {}
    for match in FROM_JOIN_RE.finditer(sql):
        table = match.group(1).lower()
        alias = (match.group(2) or "").lower()
        mapping.setdefault(table, table)
        if alias and alias not in SQL_KEYWORDS:
            mapping[alias] = table
    return mapping


def resolve(name, aliases, real_tables):
    low = name.lower()
    target = aliases.get(low, low)
    return target if target in real_tables else None


def gold_tables(sql, db_path):
    _, _, real_tables = schema_columns(db_path)
    aliases = alias_map(sql)
    tables = set()
    con = _connect(db_path)
    try:
        try:
            for row in con.execute("EXPLAIN QUERY PLAN " + sql):
                for name in PLAN_TABLE_RE.findall(str(row[-1])):
                    resolved = resolve(name, aliases, real_tables)
                    if resolved:
                        tables.add(resolved)
        except sqlite3.Error:
            pass
    finally:
        con.close()
    for match in FROM_JOIN_RE.finditer(sql):
        resolved = resolve(match.group(1), aliases, real_tables)
        if resolved:
            tables.add(resolved)
    return tables


def gold_columns(sql, db_path, tables=None):
    pairs, by_column, _ = schema_columns(db_path)
    tables = {t.lower() for t in (tables if tables is not None else gold_tables(sql, db_path))}
    text = STRING_RE.sub(" ", sql)
    found = set()
    for ident in IDENT_RE.findall(text):
        low = ident.lower()
        if low in SQL_KEYWORDS or low in tables:
            continue
        owners = by_column.get(low, set())
        if owners & tables:
            found.add((sorted(owners & tables)[0], low))
    return found


def gold_join_keys(sql, db_path, tables=None):
    """Explicit a.x = b.y pairs (aliases resolved) plus verified FK edges."""
    _, _, real_tables = schema_columns(db_path)
    aliases = alias_map(sql)
    tables = {t.lower() for t in (tables if tables is not None else gold_tables(sql, db_path))}
    keys = set()
    for a, ca, b, cb in EQ_RE.findall(sql):
        ta, tb = resolve(a, aliases, real_tables), resolve(b, aliases, real_tables)
        if ta and tb and ta in tables and tb in tables and ta != tb:
            keys.add((ta, ca.lower(), tb, cb.lower()))
    con = _connect(db_path)
    try:
        for table in tables:
            try:
                for row in con.execute(f'PRAGMA foreign_key_list("{table}")'):
                    target = str(row[2]).lower()
                    if target in tables and target != table:
                        keys.add((table, str(row[3]).lower(), target, str(row[4]).lower()))
            except sqlite3.Error:
                continue
    finally:
        con.close()
    return keys


def extract_requirements(sql, db_path):
    from eval.sql_dependencies import sqlite_requirements
    resolved = sqlite_requirements(db_path, sql)
    tables = resolved["tables"]
    return {"tables": tables, "columns": resolved["columns"],
            "resolved": resolved["ok"], "method": resolved["method"],
            **({"error": resolved["error"]} if "error" in resolved else {}),
            "join_keys_method": "legacy_explicit_name_scan_plus_declared_fk_heuristic",
            "join_keys": sorted("=".join([f"{a}.{ca}", f"{b}.{cb}"]) for a, ca, b, cb in
                                gold_join_keys(sql, db_path, tables)) if resolved["ok"] else []}
