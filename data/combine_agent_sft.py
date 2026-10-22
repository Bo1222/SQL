"""Combine local on-policy and scripted demonstrations with provenance and deduplication."""
import argparse
import hashlib
import json
import random
from collections import Counter

from agent.common import read_jsonl, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--on-policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    synthetic, policy = read_jsonl(a.synthetic), read_jsonl(a.on_policy)
    if not synthetic or not policy:
        p.error("Both sources must contain successful demonstrations; bootstrap before resampling")
    fingerprints = {r["tokenizer_sha256"] for r in synthetic + policy}
    if len(fingerprints) != 1 or any(r["split"] != "sft" for r in synthetic + policy):
        p.error("All records must use the same tokenizer and SFT split")
    rng = random.Random(a.seed)
    rng.shuffle(synthetic); rng.shuffle(policy)
    size = min(len(synthetic), len(policy))
    selected, seen = [], set()
    for row in synthetic[:size] + policy[:size]:
        key = hashlib.sha256(json.dumps(row["trajectory"], sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            selected.append({**row, "id": row["parent_id"] + "_" + key[:12]})
    rng.shuffle(selected)
    write_jsonl(a.out, selected)
    write_json(a.out + ".summary.json", {"records": len(selected), "sources": dict(Counter(r["source"] for r in selected)),
                                          "seed": a.seed, "tokenizer_sha256": fingerprints.pop()})


if __name__ == "__main__":
    main()
