"""Build the RL prompt dataset in veRL parquet format.

Prompt contains only the table-name catalog (no DDL). Gold SQL goes to
``reward_model.ground_truth`` and is never placed in the prompt.

The builder is deliberately strict about the data role:

* every source row must be ``split == "rl"``;
* RL databases/questions must not overlap val, test or dev200;
* a named runtime builds the same catalog-only prompt used by evaluation;
  the legacy E1.1 query-centric contract remains an explicit option;
* a manifest records hashes, counts and the retrieval/prompt versions so the
  parquet can be audited later.

This script writes a new parquet; it never overwrites the legacy
``artifacts/datasets/rl_full.parquet`` unless the caller explicitly chooses
that path.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.protocol import (  # noqa: E402
    build_prompt,
    catalog_from_metadata,
    prompt_version_for,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_eval_rows(path):
    path = Path(path)
    if not path.is_file():
        return None, f"missing eval file: {path}"
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read eval file {path}: {exc}"
    return rows, None


def check_disjoint(rows, eval_name, eval_rows, *, allow_db_overlap, allow_question_overlap):
    """Return (errors, warnings, summary) for one eval split."""
    errors, warnings = [], []
    train_dbs = {row["db_id"] for row in rows}
    train_ids = {row["id"] for row in rows}
    eval_dbs = {row["db_id"] for row in eval_rows}
    eval_ids = {row["id"] for row in eval_rows}
    db_overlap = sorted(train_dbs & eval_dbs)
    id_overlap = sorted(train_ids & eval_ids)
    if db_overlap and not allow_db_overlap:
        errors.append(f"RL databases overlap {eval_name}: {len(db_overlap)} e.g. {db_overlap[:5]}")
    elif db_overlap:
        warnings.append(f"RL databases overlap {eval_name} ({len(db_overlap)}), explicitly allowed")
    if id_overlap and not allow_question_overlap:
        errors.append(f"RL question ids overlap {eval_name}: {len(id_overlap)} e.g. {id_overlap[:5]}")
    elif id_overlap:
        warnings.append(f"RL question ids overlap {eval_name} ({len(id_overlap)}), explicitly allowed")
    return errors, warnings, {
        "eval": eval_name,
        "eval_rows": len(eval_rows),
        "db_overlap": len(db_overlap),
        "question_overlap": len(id_overlap),
        "overlap_dbs": db_overlap[:20],
        "overlap_ids": id_overlap[:20],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/bird_agent/rl.jsonl")
    p.add_argument("--metadata-dir", default="data/processed/bird_agent/metadata")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default="",
                   help="default: <out>.manifest.json")
    p.add_argument("--val-data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--test-data", default="data/processed/bird_agent/test.jsonl")
    p.add_argument("--dev200-data", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--prompt-mode", choices=("query", "query_v3", "sft"), default="query")
    p.add_argument("--max-tools", type=int, default=4)
    p.add_argument("--retrieval-version", default="E1.1")
    p.add_argument("--runtime-contract", choices=["", "A-e1.2-typed-v1", "A-e1.2-typed-v3"], default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-version", default="bird_rl_e1_1_v1")
    p.add_argument("--require-split", default="rl",
                   help="source split required for every row; use 'val' for dev200")
    p.add_argument("--skip-disjoint-check", action="store_true",
                   help="build an eval parquet without checking it against itself")
    p.add_argument("--allow-db-overlap", action="store_true")
    p.add_argument("--allow-question-overlap", action="store_true")
    a = p.parse_args()
    if a.runtime_contract:
        from agent.runtime_contract import contract_for
        contract = contract_for(a.runtime_contract)
        a.retrieval_version = contract["retrieval_version"]
        a.prompt_mode = contract["prompt_mode"]
        a.max_tools = contract["max_tools"]
    if a.max_tools < 1:
        p.error("--max-tools must be positive")
    if a.limit < 0:
        p.error("--limit must be non-negative")

    data_path = Path(a.data)
    if not data_path.is_file():
        raise SystemExit(f"RL source missing: {data_path}")
    rows = read_jsonl(data_path)
    if not rows:
        raise SystemExit("RL source is empty")

    errors, warnings = [], []
    ids = [row.get("id") for row in rows]
    if any(not value for value in ids):
        errors.append("some RL rows have no id")
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate RL ids: {len(ids) - len(set(ids))}")
    bad_split = [row.get("id") for row in rows if row.get("split") != a.require_split]
    if bad_split:
        errors.append(
            f"rows with split != {a.require_split!r}: {len(bad_split)} e.g. {bad_split[:5]}")
    for row in rows:
        if not row.get("question") or not row.get("gold_sql") or not row.get("db_id"):
            errors.append(f"row {row.get('id')!r} lacks question/gold_sql/db_id")
            break
        if not Path(row.get("db_path", "")).is_file():
            errors.append(f"row {row.get('id')!r} db_path missing: {row.get('db_path')}")
            break
        metadata_path = Path(row.get("metadata_path") or Path(a.metadata_dir) / f"{row['db_id']}.json")
        if not metadata_path.is_file():
            errors.append(f"row {row.get('id')!r} metadata missing: {metadata_path}")
            break

    eval_summaries = []
    if not a.skip_disjoint_check:
        for name, path in (("val", a.val_data), ("test", a.test_data), ("dev200", a.dev200_data)):
            eval_rows, warning = load_eval_rows(path)
            if eval_rows is None:
                warnings.append(warning)
                continue
            split_errors, split_warnings, summary = check_disjoint(
                rows, name, eval_rows,
                allow_db_overlap=a.allow_db_overlap,
                allow_question_overlap=a.allow_question_overlap,
            )
            errors.extend(split_errors)
            warnings.extend(split_warnings)
            eval_summaries.append(summary)

    if a.limit:
        import random
        rows = random.Random(a.seed).sample(rows, min(a.limit, len(rows)))

    metadata_cache, runtime_stores = {}, {}
    records = []
    source_hash = sha256_file(data_path)
    prompt_version = prompt_version_for(a.prompt_mode)
    for row in rows:
        db_id = row["db_id"]
        if db_id not in metadata_cache:
            path = Path(row.get("metadata_path") or Path(a.metadata_dir) / f"{db_id}.json")
            metadata_cache[db_id] = json.loads(path.read_text(encoding="utf-8"))
        catalog = catalog_from_metadata(metadata_cache[db_id])
        if a.runtime_contract:
            from agent.runtime_contract import load_store
            if db_id not in runtime_stores:
                runtime_stores[db_id] = load_store(ROOT / "artifacts/catalog", db_id, row["db_path"])
            runtime_store = runtime_stores[db_id]
            catalog = "\n".join(runtime_store.table_names)
        prompt = build_prompt(
            row["question"], row.get("evidence", ""), catalog,
            max_tools=a.max_tools, prompt_mode=a.prompt_mode,
        )
        records.append({
            "data_source": "bird_sql",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "text2sql",
            "reward_model": {"style": "rule", "ground_truth": row["gold_sql"]},
            "extra_info": {
                "question_id": row["id"],
                "db_id": db_id,
                "db_path": row["db_path"],
                "metadata_path": row.get("metadata_path", ""),
                "question": row["question"],
                "evidence": row.get("evidence", ""),
                "split": row.get("split", "rl"),
                "dataset_version": a.dataset_version,
                "prompt_mode": a.prompt_mode,
                "prompt_version": prompt_version,
                "retrieval_version": a.retrieval_version,
                "max_tools": a.max_tools,
                "source_data": str(data_path),
                "source_data_sha256": source_hash,
            },
        })
        if a.runtime_contract:
            from agent.runtime_contract import FINGERPRINTS, contract_for
            contract = contract_for(a.runtime_contract)
            records[-1]["extra_info"].update({
                "runtime_contract": a.runtime_contract,
                "runtime_fingerprint": FINGERPRINTS[a.runtime_contract],
                "score_mode": contract["score_mode"],
                "reward_contract": contract["reward_contract"],
                "observation_mode": contract["observation_mode"],
                "resolve_implicit_fks": contract["resolve_implicit_fks"],
                "runtime_config": dict(contract),
            })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        warnings.append(f"overwriting existing output: {out}")
    pd.DataFrame(records).to_parquet(out, index=False)
    manifest_path = Path(a.manifest) if a.manifest else Path(str(out) + ".manifest.json")
    manifest = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "out": str(out),
        "out_sha256": sha256_file(out),
        "rows": len(records),
        "dbs": len({record["extra_info"]["db_id"] for record in records}),
        "source_data": str(data_path),
        "source_data_sha256": source_hash,
        "dataset_version": a.dataset_version,
        "prompt_mode": a.prompt_mode,
        "prompt_version": prompt_version,
        "runtime_contract": a.runtime_contract,
        "retrieval_version": a.retrieval_version,
        "max_tools": a.max_tools,
        "limit": a.limit,
        "seed": a.seed,
        "eval_disjointness": eval_summaries,
        "prompt_chars": {
            "min": min(len(record["prompt"][0]["content"]) for record in records),
            "max": max(len(record["prompt"][0]["content"]) for record in records),
            "mean": round(sum(len(record["prompt"][0]["content"]) for record in records)
                          / max(len(records), 1), 2),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("ok", "out", "rows", "dbs", "prompt_mode", "retrieval_version")},
                     ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
