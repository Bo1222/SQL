"""E1-01: freeze the E1 runtime contract used by collection, SFT and evaluation."""
import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.protocol import INSTRUCTION, PROMPT_VERSION, SFT_INSTRUCTION, SFT_PROMPT_VERSION
from agent.sql_episode import _encode_compact, _encode_legacy, encode_observation
from e1_alignment.common import (CATALOG_DIR, DEV200, DEV200_MANIFEST, MODEL_PATH,
                                 ORIGINAL_ADAPTER, ROOT, canonical_hash, sha256_file,
                                 sha256_text, write_json)
from e1_alignment.runtime import RuntimeContract

RUNTIME_FILES = [
    "agent/protocol.py", "agent/sql_episode.py", "agent/bird_agent_loop.py",
    "retrieval/catalog_store.py", "retrieval/catalog_parser.py", "retrieval/retriever.py",
    "retrieval/e1_1.py", "retrieval/observation_formatter.py", "retrieval/schema_extractor.py",
    "tools/schema_lookup.py", "tools/sql_execute.py", "tools/sql_worker.py",
    "reward/bird_ex_reward.py", "eval/gold_extraction.py",
    "e1_alignment/runtime.py", "e1_alignment/engine.py", "e1_alignment/common.py",
]
PIPELINE_FILES = [
    "e1_alignment/fingerprint.py", "e1_alignment/collect_a.py", "e1_alignment/collect_b.py",
    "e1_alignment/collect_c.py", "e1_alignment/validate.py", "e1_alignment/sft_train.py",
    "eval/evaluate_agent.py",
]


def file_hashes(paths):
    def one(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        return {"path": str(path.relative_to(ROOT)), "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else None}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, paths))


def model_fingerprint():
    files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "merges.txt", "vocab.json", "added_tokens.json", "special_tokens_map.json",
             "chat_template.jinja", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors"]
    entries = file_hashes([MODEL_PATH / name for name in files])
    # The adapter that produced the merged checkpoint is kept as provenance only;
    # collection and training both initialize from the merged SFT capability.
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]
    adapter = file_hashes([ORIGINAL_ADAPTER / name for name in adapter_files])
    return {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "files": entries,
        "model_sha256": canonical_hash(entries),
        "original_adapter": {"path": str(ORIGINAL_ADAPTER.relative_to(ROOT)), "files": adapter,
                             "adapter_sha256": canonical_hash(adapter)},
        "tokenizer_sha256": next((e["sha256"] for e in entries if e["path"].endswith("tokenizer.json")), None),
    }


def database_fingerprint():
    manifest = json.loads((ROOT / "artifacts/manifests/catalog_manifest.json").read_text(encoding="utf-8"))
    paths = sorted({entry["db_path"] for entry in manifest["stores_detail"]})
    entries = file_hashes(paths)
    return {
        "catalog_manifest_sha256": sha256_file(ROOT / "artifacts/manifests/catalog_manifest.json"),
        "catalog_manifest_summary": {k: manifest[k] for k in
                                     ("stores", "missing_databases", "tables", "columns", "relationships",
                                      "descriptions_matched", "catalog_match_rate")},
        "databases": entries,
        "databases_sha256": canonical_hash(entries),
        "store_version": "v1",
    }


def environment_fingerprint():
    def version(name):
        try:
            module = __import__(name)
            return getattr(module, "__version__", "unknown")
        except Exception:
            return None
    packages = {name: version(name) for name in
                ("torch", "torch_npu", "vllm", "transformers", "peft", "datasets", "accelerate")}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout
        dirty_hash = sha256_text(status)
    except Exception:
        commit, dirty_hash = None, None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": packages, "git_commit": commit, "git_status_sha256": dirty_hash,
        "cann": "8.5.1", "device": "Ascend 910B4 32GB", "visible_devices": 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--skip-db-hash", action="store_true")
    p.add_argument("--retrieval-version", default="E1", choices=("E1", "E1.1"))
    a = p.parse_args()

    contract = RuntimeContract(retrieval_version=a.retrieval_version)
    code = file_hashes(RUNTIME_FILES)
    pipeline = file_hashes([name for name in PIPELINE_FILES if (ROOT / name).is_file()])
    metadata = json.loads((ROOT / "artifacts/manifests/catalog_manifest.json").read_text(encoding="utf-8"))
    dev_manifest = json.loads(DEV200_MANIFEST.read_text(encoding="utf-8"))

    fingerprint = {
        "format": "sqlrl-e1-runtime-fingerprint-v1",
        "purpose": "M6-08i fixed E1 runtime contract; all collection/SFT/evaluation arms reference this hash",
        "contract": contract.as_dict(),
        "budgets_note": "max_rounds/max_tools/max_schema_lookups/max_sql_executes are per-episode totals",
        "prompt": {
            "prompt_version": PROMPT_VERSION,
            "prompt_mode": "query",
            "instruction": INSTRUCTION,
            "instruction_sha256": sha256_text(INSTRUCTION),
            "catalog_source": "data/processed/bird_agent/metadata/<db_id>.json tables[].name",
            "catalog_manifest_tables": metadata["tables"],
            "sft_reference": {"prompt_version": SFT_PROMPT_VERSION,
                              "instruction_sha256": sha256_text(SFT_INSTRUCTION),
                              "note": "historical SFT prompt; not used by the frozen E1 runtime"},
        },
        "tool_interface": {
            "actions": ["schema_lookup", "sql_execute", "answer"],
            "schema_lookup_arguments": ["query", "table", "column"],
            "schema_lookup_requires": "query or table; all values nonempty strings",
            "sql_execute_and_answer_require": "SELECT or WITH",
            "protocol_action_tags": ["schema_lookup", "sql_execute", "answer"],
            "vllm_stop_strings": [],
            "observation_wrapper": "<observation>json</observation>\\n",
            "observation_escapes": ["<", ">"],
            "observation_mode": "compact",
            "observation_budget_tokens": contract.max_observation_tokens,
            "compact_encoder_sha256": sha256_text(encode_observation.__code__.co_code.hex()),
            "compact_encoder_source_sha256": sha256_file(ROOT / "agent/sql_episode.py"),
            "legacy_encoder_reference_sha256": sha256_text(_encode_legacy.__code__.co_code.hex()),
            "compact_impl_reference_sha256": sha256_text(_encode_compact.__code__.co_code.hex()),
            "retrieval": {"backend": f"{a.retrieval_version} unified",
                          "retrieval_version": a.retrieval_version,
                          "bm25": {"k1": 1.2, "b": 0.75},
                          "top_k_columns": 10, "max_tables": 6, "hops": 2,
                          "value_lookup": False, "elapsed_ms_visible": False,
                          "retrieval_metadata_visible": False},
        },
        "scoring": {
            "implementation": "reward/bird_ex_reward.py::score_sql",
            "timeout_s": contract.score_timeout,
            "gold_rows_vs_pred_rows": "BIRD-compatible execution comparison in tools/sql_worker.py",
            "source_sha256": sha256_file(ROOT / "reward/bird_ex_reward.py"),
        },
        "evaluation": {
            "dev_set": str(DEV200.relative_to(ROOT)),
            "dev_manifest_sha256": sha256_file(DEV200_MANIFEST),
            "dev_ids_sha256": dev_manifest.get("sha256_sorted_ids"),
            "protocol": "greedy, temperature=0, top_p=1, same E1 contract, autonomous lookup",
        },
        "collection_sampling": {
            "temperature": 1.0, "top_p": 1.0, "n_per_question": 4,
            "note": "collection sampling is recorded separately from greedy evaluation",
        },
        "code": {"runtime_files": code, "runtime_code_sha256": canonical_hash(code),
                 "pipeline_files": pipeline, "pipeline_code_sha256": canonical_hash(pipeline)},
        "environment": environment_fingerprint(),
        "model": model_fingerprint(),
        "database": ({"skipped": True} if a.skip_db_hash else database_fingerprint()),
    }
    # The canonical hash covers the fixed runtime contract, not the collection
    # pipeline scripts (which can be extended without changing the environment).
    payload = {k: v for k, v in fingerprint.items() if k not in ("fingerprint_sha256", "code")}
    payload["runtime_code"] = {"files": code,
                               "sha256": fingerprint["code"]["runtime_code_sha256"]}
    fingerprint["fingerprint_sha256"] = canonical_hash(payload)
    write_json(ROOT / a.out, fingerprint)
    summary = {"out": a.out, "fingerprint_sha256": fingerprint["fingerprint_sha256"],
               "model_sha256": fingerprint["model"]["model_sha256"],
               "code_sha256": fingerprint["code"]["runtime_code_sha256"],
               "databases_sha256": (fingerprint["database"].get("databases_sha256")
                                    if not a.skip_db_hash else None)}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
