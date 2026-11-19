"""Swappable schema_lookup backends for the E0 / E1 paired comparison.

E0 (legacy): the frozen implementation shipped with the previous project.
E1 (unified): the new Catalog + SQLite + PK/FK retrieval backend.
Everything else in the episode (protocol, SQL tool, budgets, generation) is shared.
"""
import json
import sys
from pathlib import Path

LEGACY_ROOT = Path("/home/ma-user/work/_sqlrl_legacy_20261107")
METADATA_DIR = Path("data/processed/bird_agent/metadata")


class LegacySchemaLookup:
    """E0: the previous `DatabaseTools.schema_lookup` (schema + description keyword search)."""

    name = "legacy"

    def __init__(self, row, timeout=10):
        from eval.legacy_schema_lookup import LegacySchemaLookupImpl
        metadata_path = row.get("metadata_path") or METADATA_DIR / f"{row['db_id']}.json"
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self._tools = LegacySchemaLookupImpl(metadata)

    def __call__(self, arguments):
        return self._tools.schema_lookup(arguments)


class UnifiedSchemaLookup:
    """E1: unified Catalog + Schema + PK/FK retrieval backend."""

    name = "unified"

    def __init__(self, row, catalog_dir="artifacts/catalog", token_budget=1024,
                 count_tokens=None, enable_value_lookup=False, retrieval_version="E1",
                 resolve_implicit_fks=False):
        from retrieval.catalog_store import CatalogStore
        from tools.schema_lookup import SchemaLookupTool
        if resolve_implicit_fks:
            from agent.runtime_contract import load_store
            store = load_store(catalog_dir, row["db_id"], row["db_path"])
        else:
            store = CatalogStore(Path(catalog_dir) / row["db_id"])
        self._tool = SchemaLookupTool(store, token_budget=token_budget, count_tokens=count_tokens,
                                      enable_value_lookup=enable_value_lookup,
                                      retrieval_version=retrieval_version)

    def __call__(self, arguments):
        return self._tool(arguments)


def build_backend(name, row, tokenizer=None, catalog_dir="artifacts/catalog", token_budget=1024,
                  enable_value_lookup=False, retrieval_version="E1", resolve_implicit_fks=False):
    if name == "legacy":
        return LegacySchemaLookup(row)
    if name == "unified":
        counter = (lambda text: len(tokenizer.encode(text, add_special_tokens=False))) if tokenizer else None
        return UnifiedSchemaLookup(row, catalog_dir=catalog_dir, token_budget=token_budget,
                                   count_tokens=counter, enable_value_lookup=enable_value_lookup,
                                   retrieval_version=retrieval_version, resolve_implicit_fks=resolve_implicit_fks)
    raise ValueError(f"unknown backend: {name}")

