"""Review-bound selection entry point. Refuses to write a dataset on failure."""
import argparse
from pathlib import Path
from e1_alignment.common import read_jsonl, write_json
from e1_alignment.rebuild.freeze import freeze, select
from e1_alignment.rebuild.contract import verify_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="artifacts/m6/sft_rebuild/all_audited_candidates.jsonl")
    p.add_argument("--reviews", default="reports/m6/sft_rebuild/individual_reviews.jsonl")
    p.add_argument("--report", default="reports/m6/sft_rebuild/selection_report.json")
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    if not Path(a.candidates).is_file() or not Path(a.reviews).is_file():
        raise SystemExit("candidates and actual individual reviews required")
    rows, reviews = read_jsonl(a.candidates), read_jsonl(a.reviews)
    index = {r["trajectory_id"]: r for r in reviews}
    if len(index) != len(reviews):
        raise SystemExit("duplicate_reviews")
    picked, report = select(rows, index)
    write_json(a.report, report)
    if not picked:
        raise SystemExit("dataset gate failed: " + report["status"])
    if a.freeze:
        fingerprint = verify_contract()
        freeze(picked, a.reviews, "artifacts/m6/sft_rebuild/train.jsonl",
               "artifacts/m6/sft_rebuild/frozen_manifest.json", "artifacts/m6/sft_rebuild/monitor64.jsonl", fingerprint)


if __name__ == "__main__":
    main()
