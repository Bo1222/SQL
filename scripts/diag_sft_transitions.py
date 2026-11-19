"""Transition analysis for the A/B/C diagnostic: which error class moves?"""
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def bucket(record):
    if record.get("correct"):
        return "correct"
    if record.get("valid"):
        return "semantic_wrong"
    error = record.get("error_type")
    if error == "no_such_column":
        return "schema_error"
    if error == "syntax_error":
        return "syntax_error"
    if record.get("finish_reason") != "answer_submitted":
        return "no_answer"
    return error or "other_invalid"


def main():
    base = Path("reports/m6")
    arms = {arm: {r["id"]: r for r in read_jsonl(base / f"diag_sft_{arm}.jsonl")}
            for arm in ("A", "B", "C") if (base / f"diag_sft_{arm}.jsonl").is_file()}
    for other in ("B", "C"):
        if other not in arms:
            continue
        rows = []
        for qid, record in arms["A"].items():
            if qid not in arms[other]:
                continue
            rows.append((bucket(record), qid, record, arms[other][qid]))
        print(f"=== A -> {other} ===")
        by_bucket = Counter()
        for source, qid, a, b in rows:
            by_bucket[(source, bool(b["correct"]))] += 1
        for source in ("correct", "semantic_wrong", "schema_error", "syntax_error", "no_answer", "other_invalid"):
            yes = by_bucket[(source, True)]
            total = yes + by_bucket[(source, False)]
            if total:
                print(f"{source:15s} {yes:3d}/{total:3d} correct after = {yes/total:.3f}")
        gains = sum(1 for s, q, a, b in rows if not a["correct"] and b["correct"])
        losses = sum(1 for s, q, a, b in rows if a["correct"] and not b["correct"])
        print(f"net gain={gains} loss={losses} delta={(gains-losses)/len(rows):+.4f}")
        # list a few flips for inspection
        flips = [(s, q, a, b) for s, q, a, b in rows if a["correct"] != b["correct"]]
        for source, qid, a, b in flips[:5]:
            print(" flip", source, qid, "A", a.get("error_type"), "->", other, b.get("error_type"),
                  "correct", a["correct"], "->", b["correct"])
        print()


if __name__ == "__main__":
    main()
