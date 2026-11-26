"""Deterministic split of the 31 reviewed hard questions into seen/held-out.

Seen (11) are the only questions used for Phase 1A self-sampling and SFT data.
Held-out (20) are never used for data generation, teacher prompting, or
threshold tuning; they are the hard-generalization gate after SFT.

The split is stratified by category and greedy on DB diversity.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ALLOCATION = {
    "schema_column_grounding": 2,
    "schema_plus_semantic": 3,
    "semantic_result_error": 6,
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="reports/m6/sft_first_20261120/hard_question_audit.json")
    parser.add_argument("--data", default="data/processed/bird_agent/rl_reviewed_20261118.jsonl")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120/self_sampling")
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    rows = {row["id"]: row for row in read_jsonl(args.data)}
    hard = [row for row in audit["all_zero_details"] if not row["quarantined"]]
    by_category = defaultdict(list)
    for row in hard:
        by_category[row["category"]].append(row)
    if set(by_category) != set(ALLOCATION) or len(hard) != 31:
        raise SystemExit(f"unexpected categories/count: {set(by_category)} {len(hard)}")
    seen, heldout = [], []
    for category, count in ALLOCATION.items():
        candidates = sorted(by_category[category],
                            key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        picked, used_dbs = [], set()
        for row in candidates:
            if len(picked) >= count:
                break
            if row["db_id"] not in used_dbs:
                picked.append(row)
                used_dbs.add(row["db_id"])
        for row in candidates:
            if len(picked) >= count:
                break
            if row not in picked:
                picked.append(row)
        if len(picked) != count:
            raise SystemExit(f"cannot allocate {count} for {category}")
        seen.extend(picked)
        heldout.extend([row for row in candidates if row not in picked])
    if len(seen) != 11 or len(heldout) != 20:
        raise SystemExit(f"bad split sizes: {len(seen)}/{len(heldout)}")
    if {row["id"] for row in seen} & {row["id"] for row in heldout}:
        raise SystemExit("overlap between seen and held-out")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("tasks_11_seen.jsonl", seen), ("tasks_20_heldout.jsonl", heldout)):
        tasks = [rows[row["id"]] for row in split]
        path = out / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in tasks) + "\n",
                        encoding="utf-8")
    manifest = {
        "n_seen": len(seen), "n_heldout": len(heldout),
        "allocation": ALLOCATION,
        "seen": [{"id": row["id"], "db_id": row["db_id"], "category": row["category"]}
                 for row in seen],
        "heldout": [{"id": row["id"], "db_id": row["db_id"], "category": row["category"]}
                    for row in heldout],
        "seen_dbs": len({row["db_id"] for row in seen}),
        "heldout_dbs": len({row["db_id"] for row in heldout}),
        "split_rule": "sha256(id) order within category, greedy distinct DBs, fixed 2/3/6",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
