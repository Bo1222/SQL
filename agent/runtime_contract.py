"""Opt-in A runtime shared by RL, offline calibration and evaluation."""
import hashlib
import json
from pathlib import Path

from agent.protocol import build_prompt
from agent.sql_episode import EpisodeConfig
from retrieval.catalog_store import CatalogStore
from retrieval.relationships import resolve_implicit_foreign_keys

CONTRACT_ID = "A-e1.2-typed-v1"
CONTRACT = {
    "id": CONTRACT_ID, "retrieval_version": "E1.2", "observation_mode": "compact_v2",
    "resolve_implicit_fks": True, "score_mode": "bird_set", "reward_contract": "typed-v1",
    "max_rounds": 5, "max_tools": 4, "max_schema_lookups": 2, "max_sql_executes": 2,
    "max_invalid_actions": 2, "max_action_tokens": 512, "max_observation_tokens": 1024,
    "max_context": 8192, "tool_timeout": 10, "score_timeout": 30, "preview_rows": 3,
    "max_response_tokens": 4096,
    "use_chat_template": False, "prompt_mode": "query",
}
FINGERPRINT = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def config():
    fields = EpisodeConfig.__dataclass_fields__
    return EpisodeConfig(**{key: value for key, value in CONTRACT.items() if key in fields})


def load_store(catalog_dir, db_id, db_path):
    store = CatalogStore(Path(catalog_dir) / db_id)
    repairs = resolve_implicit_foreign_keys(store, db_path)
    if repairs["unresolved"]:
        raise ValueError(f"unresolved FK endpoints in {db_id}: {repairs['unresolved']}")
    store.relationship_repairs = repairs
    _ = store.adjacency
    return store


def prompt(row, store):
    return build_prompt(row["question"], row.get("evidence", ""), "\n".join(store.table_names), 4)


def validate_extra(extra):
    for key, expected in (("runtime_contract", CONTRACT_ID), ("runtime_fingerprint", FINGERPRINT),
                          ("score_mode", "bird_set"), ("retrieval_version", "E1.2")):
        if extra.get(key) != expected:
            raise ValueError(f"runtime contract mismatch for {key}: {extra.get(key)!r} != {expected!r}")
