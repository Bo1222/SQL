"""Resolve implicit SQLite FK endpoints without changing catalog files."""
import sqlite3
from pathlib import Path


def resolve_implicit_foreign_keys(store, db_path):
    """Resolve omitted SQLite FK targets using real PK order, not a guessed join.

    SQLite permits REFERENCES parent without naming the parent PK columns.
    PRAGMA's FK group/sequence and table_info PK order also resolve composite
    keys. Non-SQLite or ambiguous missing endpoints remain explicit failures.
    """
    missing = [r for r in store.relationships
               if not r.get("from_column") or not r.get("to_column")]
    fixes, unresolved = [], []
    if not missing:
        return {"fixes": fixes, "unresolved": unresolved}
    uri = Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    resolutions = {}
    try:
        names = {t["name"].lower(): t["name"] for t in store.schema["tables"]}
        for child in store.table_names:
            groups = {}
            for fk in con.execute(f"PRAGMA foreign_key_list({quote(child)})"):
                groups.setdefault(fk[0], []).append(fk)
            for group in groups.values():
                parent = names.get(str(group[0][2]).lower())
                if parent is None:
                    continue
                info = list(con.execute(f"PRAGMA table_info({quote(parent)})"))
                pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                for fk in group:
                    target = fk[4]
                    if target is None and len(group) == len(pk) and fk[1] < len(pk):
                        target = pk[fk[1]]
                    if target:
                        key = (child.lower(), str(fk[3]).lower(), parent.lower())
                        resolutions.setdefault(key, set()).add(target)
    finally:
        con.close()
    repaired = []
    for old in store.relationships:
        rel = dict(old)
        if not rel.get("to_column") and rel.get("from_column") and rel.get("source") == "sqlite_fk":
            key = (rel["from_table"].lower(), rel["from_column"].lower(), rel["to_table"].lower())
            targets = resolutions.get(key, set())
            if len(targets) == 1:
                rel["to_column"] = next(iter(targets))
                fixes.append({"before": dict(old), "after": dict(rel)})
        if not rel.get("from_column") or not rel.get("to_column"):
            unresolved.append(dict(rel))
        repaired.append(rel)
    store.relationships = repaired
    # Keep every view of a declared FK consistent, including exact inspections.
    for table in store.schema["tables"]:
        for fk in table.get("foreign_keys", []):
            if not fk.get("target_column"):
                key = (table["name"].lower(), fk["column"].lower(), fk["target_table"].lower())
                targets = resolutions.get(key, set())
                if len(targets) == 1:
                    fk["target_column"] = next(iter(targets))
    store.fk_by_table = {t["name"]: t.get("foreign_keys", []) for t in store.schema["tables"]}
    store._adjacency = None
    return {"fixes": fixes, "unresolved": unresolved}

