"""Shared CREATE TABLE DDL builder for Spider/BIRD table metadata."""
from collections import defaultdict


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_schema(entry: dict) -> str:
    tables = entry["table_names_original"]
    columns = entry["column_names_original"]
    types = entry["column_types"]
    primary_keys = entry["primary_keys"]
    foreign_keys = entry["foreign_keys"]

    by_table = defaultdict(list)
    for idx, (table_idx, name) in enumerate(columns):
        if table_idx < 0:
            continue
        by_table[table_idx].append((name, types[idx]))

    pk_by_table = defaultdict(list)
    for idx in primary_keys:
        indices = idx if isinstance(idx, (list, tuple)) else [idx]
        for sub_idx in indices:
            table_idx, name = columns[sub_idx]
            pk_by_table[table_idx].append(name)

    fk_by_table = defaultdict(list)
    for src, dst in foreign_keys:
        src_table, src_col = columns[src]
        dst_table, dst_col = columns[dst]
        fk_by_table[src_table].append((src_col, tables[dst_table], dst_col))

    statements = []
    for table_idx, table_name in enumerate(tables):
        parts = [f"  {quote(name)} {ctype}" for name, ctype in by_table[table_idx]]
        if pk_by_table[table_idx]:
            cols = ", ".join(quote(c) for c in pk_by_table[table_idx])
            parts.append(f"  PRIMARY KEY ({cols})")
        for src_col, dst_table, dst_col in fk_by_table[table_idx]:
            parts.append(
                f"  FOREIGN KEY ({quote(src_col)}) REFERENCES {quote(dst_table)}({quote(dst_col)})"
            )
        statements.append(f"CREATE TABLE {quote(table_name)} (\n" + ",\n".join(parts) + "\n);")
    return "\n\n".join(statements)
