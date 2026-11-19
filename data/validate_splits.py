"""Database-level split audit and leakage check (plan §4.3)."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows):
    return {
        "n": len(rows),
        "ids": {r["id"] for r in rows if r.get("id")},
        "dbs": {r["db_id"] for r in rows if r.get("db_id")},
        "db_counts": Counter(r.get("db_id") for r in rows if r.get("db_id")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="data/processed/bird_agent/full_sft_20261026/full_tool.jsonl")
    p.add_argument("--rl", default="data/processed/bird_agent/rl.jsonl")
    p.add_argument("--val", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--test", default="data/processed/bird_agent/test.jsonl")
    p.add_argument("--out", default="artifacts/manifests/split_audit.json")
    a = p.parse_args()

    splits = {
        "sft": summarize(read_jsonl(a.sft)),
        "rl": summarize(read_jsonl(a.rl)),
        "val": summarize(read_jsonl(a.val)),
        "test": summarize(read_jsonl(a.test)),
    }
    names = list(splits)
    db_overlap = {}
    id_overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            db_overlap[f"{left}∩{right}"] = sorted(splits[left]["dbs"] & splits[right]["dbs"])
            id_overlap[f"{left}∩{right}"] = len(splits[left]["ids"] & splits[right]["ids"])

    report = {
        "format": "bird-split-audit-v1",
        "splits": {name: {"n": s["n"], "dbs": len(s["dbs"]),
                          "db_counts": dict(sorted(s["db_counts"].items()))} for name, s in splits.items()},
        "db_overlap": {k: {"count": len(v), "dbs": v} for k, v in db_overlap.items()},
        "id_overlap": id_overlap,
        "leakage_checks": {
            "sft_vs_val_disjoint": not db_overlap["sft∩val"],
            "sft_vs_test_disjoint": not db_overlap["sft∩test"],
            "rl_vs_val_disjoint": not db_overlap["rl∩val"],
            "rl_vs_test_disjoint": not db_overlap["rl∩test"],
            "val_vs_test_disjoint": not db_overlap["val∩test"],
        },
        "notes": [
            "SFT covers every RL database; no model-level unseen database exists for RL.",
            "val/test databases are unseen at the model level (no SFT or RL overlap).",
            "Usage history: val100 subset selected SFT checkpoints; val590 evaluated once for SFT; "
            "D100 (val590 subset) was reused for RL development; test(dev1534) evaluated once for SFT; "
            "H200 evaluated once for B+100. Final confirmation must fix its protocol.",
        ],
        "audit_sha256": hashlib.sha256(json.dumps({n: s["n"] for n, s in splits.items()},
                                                  sort_keys=True).encode()).hexdigest()[:16],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"leakage_checks": report["leakage_checks"],
                      "db_overlap": {k: v["count"] for k, v in report["db_overlap"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
