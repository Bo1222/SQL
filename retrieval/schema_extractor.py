"""Extract authoritative structure from a SQLite database."""
import json
import sqlite3
from pathlib import Path


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def extract_schema(db_path):
    """Return {db_id, tables[], relationships[]} read from the real SQLite file."""
    db_path = Path(db_path).resolve()
    con = sqlite3.connect(db_path.as_uri() + "?mode=ro&immutable=1", uri=True)
    tables, relationships = [], []
    try:
        rows = list(con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"))
        for name, ddl in rows:
            info = list(con.execute(f"PRAGMA table_info({quote(name)})"))
            fks = list(con.execute(f"PRAGMA foreign_key_list({quote(name)})"))
            columns = [{"name": r[1], "type": r[2] or "", "primary_key": bool(r[5]), "not_null": bool(r[3])}
                       for r in info]
            tables.append({
                "name": name,
                "ddl": ddl or "",
                "columns": columns,
                "primary_keys": [c["name"] for c in columns if c["primary_key"]],
                "foreign_keys": [{"column": r[3], "target_table": r[2], "target_column": r[4]} for r in fks],
            })
            for r in fks:
                relationships.append({
                    "from_table": name, "from_column": r[3],
                    "to_table": r[2], "to_column": r[4],
                    "source": "sqlite_fk", "confidence": 1.0,
                })
    finally:
        con.close()
    table_names = {t["name"] for t in tables}
    for t in tables:  # drop FK targets that do not exist as tables
        t["foreign_keys"] = [f for f in t["foreign_keys"] if f["target_table"] in table_names]
    relationships = [r for r in relationships if r["to_table"] in table_names]
    return {"db_id": db_path.stem, "db_path": str(db_path), "tables": tables, "relationships": relationships}


def table_ddl(schema, table):
    for t in schema["tables"]:
        if t["name"] == table:
            return t["ddl"]
    return ""


def iter_columns(schema):
    for t in schema["tables"]:
        for c in t["columns"]:
            yield t["name"], c


if __name__ == "__main__":
    import sys
    print(json.dumps(extract_schema(sys.argv[1])["db_id"]))
