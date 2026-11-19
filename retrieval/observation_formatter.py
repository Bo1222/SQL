"""Render the unified Observation and enforce the token budget.

Degradation ladder (never drops join keys first):
  full (types + descriptions + relationships)
  -> drop descriptions
  -> trim low-score columns per table
  -> drop lowest-score tables that are not join endpoints
  -> line-level truncation as the last resort
"""
DEFAULT_CHARS_PER_TOKEN = 3.2


def default_count_tokens(text):
    return max(1, int(len(text) / DEFAULT_CHARS_PER_TOKEN)) if text else 0


def _columns_by_table(store, selected_columns):
    grouped = {}
    for record in selected_columns:
        grouped.setdefault(record["table"], []).append(record)
    for table in grouped:
        grouped[table].sort(key=lambda c: (-c.get("score", 0.0), c.get("join_key", False) is False))
    return grouped


def _render(store, tables, grouped, relationships, values, with_description=True, per_table_limit=None):
    lines = ["Relevant Database Context", "", "Tables:"]
    for table in tables:
        lines.append(f"- {table}")
        columns = grouped.get(table, [])
        if per_table_limit:
            columns = columns[:per_table_limit]
        for record in columns:
            column = record["column"]
            dtype = store.column_types.get((table, column), "") or ""
            tag = " [PK]" if store.catalog_by_column.get((table, column), {}).get("primary_key") else ""
            key = " [JOIN KEY]" if record.get("join_key") else ""
            lines.append(f"  - {column} {dtype}{tag}{key}".rstrip())
            if with_description:
                description = store.catalog_by_column.get((table, column), {}).get("description", "")
                if description:
                    lines.append(f"    Description: {description[:180]}")
    lines.append("")
    lines.append("Relationships:")
    if relationships:
        for rel in relationships:
            lines.append(f"- {rel['from_table']}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}"
                         f"   [{rel.get('source', 'sqlite_fk')}]")
    else:
        lines.append("- (no confirmed join relationship found)")
    lines.append("")
    lines.append("Relevant Values:")
    if values:
        for hit in values:
            lines.append(f"- {hit['table']}.{hit['column']} contains a value like '{hit['value']}'")
    else:
        lines.append("- (value lookup disabled or no specific entity values required)")
    return "\n".join(lines)


def format_observation(store, tables, selected_columns, relationships, values,
                       token_budget=1536, count_tokens=None):
    count_tokens = count_tokens or default_count_tokens
    grouped = _columns_by_table(store, selected_columns)
    join_tables = set()
    for rel in relationships:
        join_tables.add(rel["from_table"])
        join_tables.add(rel["to_table"])

    text = _render(store, tables, grouped, relationships, values, with_description=True)
    if count_tokens(text) <= token_budget:
        return text

    text = _render(store, tables, grouped, relationships, values, with_description=False)
    if count_tokens(text) <= token_budget:
        return text

    for limit in (6, 4, 3):
        text = _render(store, tables, grouped, relationships, values, with_description=False, per_table_limit=limit)
        if count_tokens(text) <= token_budget:
            return text

    # drop low-score tables that are not join endpoints, largest first
    keep = [t for t in tables]
    while count_tokens(text) > token_budget and len(keep) > 1:
        drop = None
        for table in reversed(keep):
            if table not in join_tables:
                drop = table
                break
        if drop is None:
            break
        keep.remove(drop)
        text = _render(store, keep, grouped, relationships, values, with_description=False, per_table_limit=3)
    if count_tokens(text) <= token_budget:
        return text

    # last resort: line truncation, keep header
    lines = text.splitlines()
    while lines and count_tokens("\n".join(lines)) > token_budget:
        lines.pop()
    lines.append("- (context truncated to fit the observation budget)")
    return "\n".join(lines)
