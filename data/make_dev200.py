"""M6-00: fix the dev200 development subset and record its hash.

Stratified by database over val.jsonl (590 questions, 7 DBs), seed=0,
proportional quotas with deterministic remainder assignment.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--manifest", default="artifacts/manifests/dev200.json")
    a = p.parse_args()

    rows = read_jsonl(a.data)
    groups = defaultdict(list)
    for row in rows:
        groups[row["db_id"]].append(row)

    quotas = {db: int(len(v) * a.limit / len(rows)) for db, v in groups.items()}
    remainder = a.limit - sum(quotas.values())
    fractional = sorted(groups, key=lambda db: (-(len(groups[db]) * a.limit / len(rows) - quotas[db]), db))
    for db in fractional[:remainder]:
        quotas[db] += 1

    rng = random.Random(a.seed)
    picked = []
    for db in sorted(groups):
        pool = sorted(groups[db], key=lambda r: str(r["id"]))
        picked.extend(rng.sample(pool, quotas[db]))
    picked.sort(key=lambda r: str(r["id"]))

    ids = [row["id"] for row in picked]
    digest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in picked) + "\n", encoding="utf-8")

    manifest = {
        "format": "dev200-v1",
        "source": a.data,
        "n": len(picked),
        "seed": a.seed,
        "per_db": {db: quotas[db] for db in sorted(quotas)},
        "source_n": len(rows),
        "source_dbs": len(groups),
        "sha256_sorted_ids": digest,
        "ids": ids,
        "note": "development/selection set; do not use as a blind test set",
    }
    mpath = Path(a.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n", "seed", "per_db", "sha256_sorted_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
