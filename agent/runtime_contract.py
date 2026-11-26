"""Opt-in A runtime shared by RL, offline calibration and evaluation.

One registry serves v1 (the frozen query prompt) and v3 (the exact-lookup
prompt).  ``CONTRACT_ID``/``FINGERPRINT`` remain v1 aliases so existing
importers and the v1 dataset bytes keep their original meaning.
"""
import hashlib
import json
from pathlib import Path

from agent.protocol import build_prompt, prompt_version_for
from agent.sql_episode import EpisodeConfig
from retrieval.catalog_store import CatalogStore
from retrieval.relationships import resolve_implicit_foreign_keys

CONTRACT_V1 = {
    "id": "A-e1.2-typed-v1", "retrieval_version": "E1.2", "observation_mode": "compact_v2",
    "resolve_implicit_fks": True, "score_mode": "bird_set", "reward_contract": "typed-v1",
    "max_rounds": 5, "max_tools": 4, "max_schema_lookups": 2, "max_sql_executes": 2,
    "max_invalid_actions": 2, "max_action_tokens": 512, "max_observation_tokens": 1024,
    "max_context": 8192, "tool_timeout": 10, "score_timeout": 30, "preview_rows": 3,
    "max_response_tokens": 4096,
    "use_chat_template": False, "prompt_mode": "query",
}
CONTRACT_V3 = {**CONTRACT_V1, "id": "A-e1.2-typed-v3", "prompt_mode": "query_v3"}
CONTRACTS = {CONTRACT_V1["id"]: CONTRACT_V1, CONTRACT_V3["id"]: CONTRACT_V3}

# v1 aliases: do not change these names or values.
CONTRACT_ID = CONTRACT_V1["id"]
CONTRACT = CONTRACT_V1


def _fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


FINGERPRINT_V1 = _fingerprint(CONTRACT_V1)
FINGERPRINT_V3 = _fingerprint(CONTRACT_V3)
FINGERPRINT = FINGERPRINT_V1
FINGERPRINTS = {CONTRACT_V1["id"]: FINGERPRINT_V1, CONTRACT_V3["id"]: FINGERPRINT_V3}


def contract_for(contract_id=None):
    contract_id = contract_id or CONTRACT_ID
    try:
        return CONTRACTS[contract_id]
    except KeyError as exc:
        raise ValueError(f"unknown runtime contract {contract_id!r}") from exc


def config(contract_id=None):
    contract = contract_for(contract_id)
    fields = EpisodeConfig.__dataclass_fields__
    return EpisodeConfig(**{key: value for key, value in contract.items() if key in fields})


def load_store(catalog_dir, db_id, db_path):
    store = CatalogStore(Path(catalog_dir) / db_id)
    repairs = resolve_implicit_foreign_keys(store, db_path)
    if repairs["unresolved"]:
        raise ValueError(f"unresolved FK endpoints in {db_id}: {repairs['unresolved']}")
    store.relationship_repairs = repairs
    _ = store.adjacency
    return store


def prompt_for(extra, store, contract_id=None):
    contract = contract_for(contract_id)
    return build_prompt(extra["question"], extra.get("evidence", ""),
                        "\n".join(store.table_names), contract["max_tools"],
                        prompt_mode=contract["prompt_mode"])


def prompt(extra, store, contract_id=None):
    """Backward-compatible alias for prompt_for."""
    return prompt_for(extra, store, contract_id)


def validate_extra(extra, contract_id=None):
    contract = contract_for(contract_id)
    expected = {
        "runtime_contract": contract["id"],
        "runtime_fingerprint": FINGERPRINTS[contract["id"]],
        "score_mode": contract["score_mode"],
        "retrieval_version": contract["retrieval_version"],
    }
    # Newer datasets carry these; older v1 fixtures may omit them. Validate
    # them whenever present so v3 cannot silently run a v1 observation mode.
    optional = {
        "observation_mode": contract["observation_mode"],
        "reward_contract": contract["reward_contract"],
        "prompt_mode": contract["prompt_mode"],
        "prompt_version": prompt_version_for(contract["prompt_mode"]),
    }
    for key, value in optional.items():
        if key in extra:
            expected[key] = value
    for key, value in expected.items():
        if extra.get(key) != value:
            raise ValueError(f"runtime contract mismatch for {key}: {extra.get(key)!r} != {value!r}")
    if "runtime_config" in extra and extra["runtime_config"] != contract:
        raise ValueError("runtime contract config mismatch")
