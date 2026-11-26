"""Reproduce the CPU-now v3 evidence with per-record provenance.

Writes under `<report>/reproduction/`:
  prompt_checks.json          tokenizer identity, token stats, v1 parity
  seed_delivery_records.jsonl one line per prepared seed and arm
  seed_delivery_summary.json  aggregate delivery/inspection/singleton
  run_info.json               python/transformers/tokenizer versions

This script is read-only over the datasets and catalogs; it writes only inside
the report directory. It never launches a model or a GPU job.
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.protocol import build_prompt  # noqa: E402
from agent.runtime_contract import FINGERPRINT_V1  # noqa: E402
from retrieval.catalog_store import CatalogStore  # noqa: E402
from tools.schema_lookup import SchemaLookupTool  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                          encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_stats(lengths):
    lengths = sorted(lengths)
    n = len(lengths)
    return {
        "n": n,
        "max": lengths[-1] if n else None,
        "p99": lengths[int(0.99 * (n - 1))] if n else None,
        "mean": round(sum(lengths) / n, 2) if n else None,
        "over_2000": sum(value > 2000 for value in lengths),
        "over_2048": sum(value > 2048 for value in lengths),
        "method": "len(tokenizer.encode(text, add_special_tokens=False)); "
                  "p99 index=int(0.99*(n-1)) on ascending lengths",
    }


def prompt_checks(model, train_path, val_path, v1_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    result = {"tokenizer_class": type(tokenizer).__name__,
              "tokenizer_name_or_path": str(model), "checks": {}}
    for name, path in (("v3_train", train_path), ("v3_dev200", val_path), ("v1_train", v1_path)):
        import pandas as pd
        frame = pd.read_parquet(path)
        lengths = [len(tokenizer.encode(item[0]["content"], add_special_tokens=False))
                   for item in frame["prompt"].tolist()]
        result["checks"][name] = {**token_stats(lengths), "path": str(path),
                                  "sha256": sha256_file(path)}
    import pandas as pd
    v1 = pd.read_parquet(v1_path)
    metadata_cache, mismatches = {}, 0
    for _, row in v1.iterrows():
        info = dict(row["extra_info"])
        db_id = info["db_id"]
        if db_id not in metadata_cache:
            metadata_path = Path(info.get("metadata_path") or
                                 f"data/processed/bird_agent/metadata/{db_id}.json")
            metadata_cache[db_id] = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = build_prompt(info["question"], info.get("evidence", ""),
                              "\n".join(t["name"] for t in metadata_cache[db_id]["tables"]),
                              4, prompt_mode="query")
        if prompt != row["prompt"][0]["content"]:
            mismatches += 1
    fingerprints = sorted({dict(info).get("runtime_fingerprint") for info in v1["extra_info"]})
    result["v1_parity"] = {
        "rows": len(v1), "prompt_mismatches": mismatches,
        "fingerprint": FINGERPRINT_V1,
        "fingerprint_matches": fingerprints == [FINGERPRINT_V1],
        "observed_fingerprints": fingerprints,
    }
    return result


def seed_delivery(report, data_path, catalog_dir):
    seeds = [row for row in read_jsonl(report / "forced_exact_dev200" / "seeds.jsonl")
             if row.get("seed")]
    rows = {row["id"]: row for row in read_jsonl(data_path)}
    stores = {}

    def store(db_id):
        if db_id not in stores:
            stores[db_id] = CatalogStore(Path(catalog_dir) / db_id)
        return stores[db_id]

    records, aggregate = [], {}
    for arm, version in (("E12_exact", "E1.2"), ("E11_ignored", "E1.1")):
        counts = Counter()
        for seed_row in seeds:
            row = rows[seed_row["id"]]
            result = SchemaLookupTool(store(row["db_id"]), token_budget=1024,
                                      retrieval_version=version)(seed_row["seed"])
            meta = result.get("retrieval_metadata") or {}
            matched = meta.get("matched_columns") or []
            singleton = (len(matched) == 1 and isinstance(matched[0], dict)
                         and str(matched[0].get("table", "")).lower()
                         == seed_row["seed"]["table"].lower()
                         and str(matched[0].get("column", "")).lower()
                         == seed_row["seed"]["column"].lower())
            record = {
                "id": seed_row["id"], "arm": arm,
                "status": result.get("status"),
                "inspection": meta.get("inspection") is True,
                "selected_columns": int(meta.get("selected_columns") or 0),
                "matched_count": len(matched),
                "singleton_match": singleton,
                "observation_chars": len(str(result.get("observation") or "")),
                "seed": seed_row["seed"],
            }
            records.append(record)
            counts["n"] += 1
            counts["ok"] += record["status"] == "ok"
            counts["inspection"] += record["inspection"] and record["selected_columns"] > 0
            counts["singleton"] += singleton
        aggregate[arm] = dict(counts)
    write_jsonl(report / "reproduction" / "seed_delivery_records.jsonl", records)
    (report / "reproduction" / "seed_delivery_summary.json").write_text(
        json.dumps(aggregate, indent=1) + "\n", encoding="utf-8")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/m6/harness_v3_20261120")
    parser.add_argument("--model",
                        default="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model")
    parser.add_argument("--data", default="artifacts/datasets/dev200.jsonl")
    parser.add_argument("--catalog-dir", default="artifacts/catalog")
    parser.add_argument("--train", default="artifacts/datasets/rl_A_typed_v3_reviewed.parquet")
    parser.add_argument("--val", default="artifacts/datasets/dev200_A_typed_v3.parquet")
    parser.add_argument("--v1", default="artifacts/datasets/rl_A_typed_v1.parquet")
    args = parser.parse_args()
    report = Path(args.report_dir)
    reproduction = report / "reproduction"
    reproduction.mkdir(parents=True, exist_ok=True)
    try:
        import transformers
        versions = {"python": platform.python_version(),
                    "transformers": transformers.__version__}
    except Exception:  # noqa: BLE001
        versions = {"python": platform.python_version()}
    checks = prompt_checks(args.model, args.train, args.val, args.v1)
    (reproduction / "prompt_checks.json").write_text(
        json.dumps(checks, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    delivery = seed_delivery(report, args.data, args.catalog_dir)
    (reproduction / "run_info.json").write_text(
        json.dumps({"versions": versions, "model": args.model, "data": args.data,
                    "train": args.train, "val": args.val, "v1": args.v1},
                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_checks": checks["checks"],
                      "v1_parity": checks["v1_parity"],
                      "seed_delivery": delivery}, ensure_ascii=False))


if __name__ == "__main__":
    main()
