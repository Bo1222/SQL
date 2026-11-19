"""Physical SQLite read dependencies, resolved without executing the query."""
from pathlib import Path
import sqlite3


def sqlite_requirements(db_path, sql):
    """Compile EXPLAIN and collect SQLite READ authorizations.

    Quoted identifiers, aliases, CTEs and subqueries are resolved by SQLite,
    rather than guessed by an identifier regex. COUNT(*) still reads a table
    even when SQLite reports an empty column name. This describes dependencies,
    not semantic correctness, discoverability or complete join semantics.
    """
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    tables, columns = set(), set()
    try:
        con.execute("PRAGMA query_only=ON")
        schema = {}
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')"):
            if table.startswith("sqlite_"):
                continue
            quoted = table.replace('"', '""')
            schema[table.lower()] = {str(r[1]).lower(): r[2] for r in con.execute(
                f'PRAGMA table_xinfo("{quoted}")') if r[6] != 1}

        def authorize(operation, a, b, database, source):
            if operation == sqlite3.SQLITE_READ and a and a.lower() in schema:
                tables.add(a.lower())
                if b:
                    columns.add(f"{a.lower()}.{b.lower()}")
            return sqlite3.SQLITE_OK

        con.set_authorizer(authorize)
        con.execute("EXPLAIN " + sql).fetchall()
        return {"ok": True, "tables": sorted(tables), "columns": sorted(columns),
                "method": "sqlite_authorizer_prepare", "schema": schema}
    except (sqlite3.Error, ValueError, TypeError) as exc:
        return {"ok": False, "tables": [], "columns": [], "error": str(exc),
                "method": "sqlite_authorizer_prepare", "schema": {}}
    finally:
        con.close()
