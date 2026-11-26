"""Export execution-verified self-sampled hard-question trajectories for SFT.

Input: records.jsonl + manifest.json from
`scripts/eval_rl_learning_pool.py` on the 11 seen reviewed hard questions
(the 31 reviewed set split into 11 seen / 20 held-out).

Output:
  sft_correct.jsonl        correct trajectories, training_eligible=True
  yield_summary.json       per-question correct/valid/all-zero counts
  YIELD.md                 human-readable yield report

The raw sampling records remain diagnostic-only; this script derives a separate
SFT data file and never mutates the benchmark or the raw records.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def error_type(record):
    score = record.get("score") or {}
    return score.get("error_type") or record.get("error_type")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-question", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--split-name", default="hard_reviewed_11_seen",
                        help="label for the seen hard split used for SFT")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    records = read_jsonl(args.records)
    by_question = defaultdict(list)
    for record in records:
        by_question[record["id"]].append(record)
    n_tasks = manifest["n_questions"]
    if len(by_question) != n_tasks:
        raise SystemExit(f"expected {n_tasks} questions, found {len(by_question)}")
    for qid, rows in by_question.items():
        if len(rows) != args.min_samples:
            raise SystemExit(f"{qid}: {len(rows)} samples, expected {args.min_samples}")
        if sorted(row["sample_index"] for row in rows) != list(range(args.min_samples)):
            raise SystemExit(f"{qid}: sample indices are not complete")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected, per_question = [], []
    for qid, rows in by_question.items():
        correct = [row for row in rows
                   if row.get("correct") and row.get("reward_available", True)]
        # Keep distinct correct SQL strings, up to the cap, in sample order.
        seen, chosen = set(), []
        for row in sorted(correct, key=lambda item: item["sample_index"]):
            sql = " ".join(str(row.get("final_sql") or "").split()).lower()
            if sql and sql not in seen:
                seen.add(sql)
                chosen.append(row)
            if len(chosen) >= args.max_per_question:
                break
        for row in chosen:
            exported = {
                "id": row["id"], "db_id": row["db_id"],
                "question": row.get("question"),
                # final_sql is the only executable target; no Gold SQL/metadata.
                "final_sql": row["final_sql"],
                "segments": row.get("segments"), "tool_events": row.get("tool_events"),
                "executions": row.get("executions"),
                "sample_index": row["sample_index"],
                "runtime_contract": manifest.get("runtime_contract"),
                "runtime_fingerprint": manifest.get("runtime_fingerprint"),
                "source": "self_rejection_phase1a",
                "sft_split": args.split_name,
                "training_eligible": True,
                "diagnostic_only": False,
            }
            forbidden = {"gold_sql", "gold_rows", "gold_error"}

            def has_forbidden(value):
                if isinstance(value, dict):
                    if forbidden & set(value):
                        return True
                    return any(has_forbidden(item) for item in value.values())
                if isinstance(value, list):
                    return any(has_forbidden(item) for item in value)
                return False

            if has_forbidden(exported):
                raise SystemExit(f"Gold-derived field in SFT row: {row['id']}")
            selected.append(exported)
        per_question.append({
            "id": qid, "db_id": rows[0]["db_id"],
            "samples": len(rows), "correct": len(correct),
            "pass_at_1": round(len(correct) / len(rows), 4),
            "pass_at_k": bool(correct),
            "valid": sum(bool(row.get("valid")) for row in rows),
            "all_zero": not correct,
            "error_types": dict(Counter(str(error_type(row)) for row in rows)),
            "selected_for_sft": len(chosen),
        })
    selected_path = out / "sft_correct.jsonl"
    selected_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in selected) + "\n",
        encoding="utf-8")
    summary = {
        "questions": n_tasks,
        "samples_per_question": args.min_samples,
        "raw_records": len(records),
        "correct_trajectories_total": sum(row["correct"] for row in per_question),
        "all_zero_questions": sum(row["all_zero"] for row in per_question),
        "mixed_or_solved_questions": sum(not row["all_zero"] for row in per_question),
        "selected_for_sft": len(selected),
        "max_per_question": args.max_per_question,
        "per_question": per_question,
        "raw_manifest": manifest,
        "sft_data_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
    }
    (out / "yield_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Phase 1A self-sampling yield", "",
        f"- Questions: {n_tasks}; samples/question: {args.min_samples}; raw records: {len(records)}",
        f"- Correct trajectories: {summary['correct_trajectories_total']}",
        f"- All-zero after sampling: {summary['all_zero_questions']}",
        f"- Selected for SFT (cap {args.max_per_question}/question): {len(selected)}",
        "", f"| id | db | correct/{args.min_samples} | valid | all-zero | selected |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in sorted(per_question, key=lambda item: (item["correct"], item["id"])):
        lines.append(f"| {row['id']} | {row['db_id']} | {row['correct']}/{row['samples']} | "
                     f"{row['valid']} | {row['all_zero']} | {row['selected_for_sft']} |")
    lines.append("")
    (out / "YIELD.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "questions", "raw_records", "correct_trajectories_total",
        "all_zero_questions", "selected_for_sft")}, indent=1))


if __name__ == "__main__":
    main()
