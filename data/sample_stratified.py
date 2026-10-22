"""Deterministic stratified subset by db_id, with a manifest for reproducibility."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def allocate(counts: dict[str, int], limit: int) -> dict[str, int]:
    limit = min(limit, sum(counts.values()))
    if limit <= len(counts):
        top = sorted(counts, key=lambda db: (-counts[db], db))[:limit]
        return {db: 1 for db in top}
    total = sum(counts.values())
    exact = {db: limit * n / total for db, n in counts.items()}
    alloc = {db: max(1, int(value)) for db, value in exact.items()}
    while sum(alloc.values()) > limit:
        db = max((d for d in alloc if alloc[d] > 1), key=lambda d: alloc[d])
        alloc[db] -= 1
    remainders = sorted(exact, key=lambda d: (exact[d] - int(exact[d])), reverse=True)
    cursor = 0
    while sum(alloc.values()) < limit:
        db = remainders[cursor % len(remainders)]
        if alloc[db] < counts[db]:
            alloc[db] += 1
        cursor += 1
    return alloc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/spider/test.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/processed/spider/test_200.jsonl")
    parser.add_argument("--manifest", default="reports/spider_test200_manifest.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_db = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)
    counts = {db: len(items) for db, items in by_db.items()}
    alloc = allocate(counts, args.limit)

    rng = random.Random(args.seed)
    chosen = []
    print(f"[manifest] source={args.data} total={len(rows)} dbs={len(by_db)} limit={args.limit} seed={args.seed}")
    for db in sorted(by_db):
        take_n = alloc.get(db, 0)
        if take_n == 0:
            continue
        items = list(by_db[db])
        rng.shuffle(items)
        take = items[:take_n]
        chosen.extend(take)
        print(f"[manifest] db={db:<28} available={len(items):>4} selected={len(take):>3}")

    chosen.sort(key=lambda row: row["id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": args.data,
        "total": len(rows),
        "selected": len(chosen),
        "limit": args.limit,
        "seed": args.seed,
        "db_counts": {db: alloc[db] for db in sorted(alloc)},
        "ids": [row["id"] for row in chosen],
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {out} and {args.manifest}")


if __name__ == "__main__":
    main()
