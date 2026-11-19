"""Fixed-weight input ablations; no Gold-dependent prompt construction.

All changes are opt-in. Existing catalog files, prompts and evaluations keep
their historical defaults. Stores are repaired in memory for this experiment.
"""
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from agent.protocol import build_prompt
from agent.sql_episode import EpisodeConfig

ARMS = ("A", "B", "C")
VERSION = "3b-input-abc-v1"
ANSWER_ONLY_INSTRUCTION = """You are a read-only SQLite database agent.
Write exactly one SQLite SELECT or WITH query that answers the question.
Use only tables and columns from the schema below.
No tools are available. Return exactly one <answer>YOUR SQL</answer> and nothing else.
"""


from retrieval.relationships import resolve_implicit_foreign_keys


def full_schema_context(store):
    """All catalog columns, using the teacher fullschema rendering limits.

    Field/value descriptions are limited to 200/160 characters, exactly as in
    that teacher arm. No tables/columns are selected using the question or Gold.
    """
    lines = [f"Database: {store.manifest.get('db_id', '')}", "Tables:"]
    for table in store.schema["tables"]:
        lines.append(f"- {table['name']}")
        for column in store.catalog:
            if column["table"] != table["name"]:
                continue
            pk = " [PK]" if column.get("primary_key") else ""
            line = f"  - {column['column']} {column.get('data_type', '')}{pk}"
            if column.get("description"):
                line += " -- " + column["description"][:200]
            if column.get("value_description"):
                line += " | values: " + column["value_description"][:160]
            lines.append(line)
    lines.append("Relationships:")
    for rel in store.relationships:
        if not rel.get("from_column") or not rel.get("to_column"):
            raise ValueError("unresolved relationship endpoint in full schema")
        lines.append(f"- {rel['from_table']}.{rel['from_column']} = "
                     f"{rel['to_table']}.{rel['to_column']} [{rel.get('source', '')}]")
    return "\n".join(lines).replace("<", "\\u003c").replace(">", "\\u003e")


def episode_config(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    config = EpisodeConfig(max_rounds=5, max_context=8192, observation_mode="compact_v2")
    if arm == "C":
        config = replace(config, max_rounds=1, max_tools=0, max_schema_lookups=0,
                         max_sql_executes=0, max_invalid_actions=1)
    return config


def input_prompt(row, store, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    catalog = "\n".join(store.table_names)
    base = build_prompt(row["question"], row.get("evidence", ""), catalog, 4)
    if arm == "A":
        return base
    schema = full_schema_context(store)
    if arm == "B":
        prefix, trajectory = base.rsplit("### Trajectory\n", 1)
        return prefix + "### Database schema\n" + schema + "\n\n### Trajectory\n" + trajectory
    return (ANSWER_ONLY_INSTRUCTION + "\n### Table catalog\n" + catalog
            + "\n### Database schema\n" + schema
            + "\n### Question\n" + row["question"]
            + "\n### Evidence\n" + (row.get("evidence") or "(not supplied)")
            + "\n### Trajectory\n")
