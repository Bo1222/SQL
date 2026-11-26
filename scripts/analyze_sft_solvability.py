"""CPU audit: which questions in the SFT training pool are solvable at n4.

Inputs:
  reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl  (4,190 questions x 4)
  artifacts/m6/e1_student_only/sft_student_full_prefer.jsonl (3,148 rows / 2,372 q)

Classifies each pool question:
  all_zero       0/4 correct (unanswerable at n4)
  first_pass     sample_index 0 correct
  later_pass_only sample 0 wrong, at least one of samples 1-3 correct

The SFT training file contains only successful trajectories, so it has no
all-zero questions by construction. The 1,815 all-zero pool questions are the
excluded/unanswerable set.

No model, no GPU, no API.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts",
                        default="reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl")
    parser.add_argument("--sft",
                        default="artifacts/m6/e1_student_only/sft_student_full_prefer.jsonl")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sft_qids, sft_rows, sft_selected_indices = set(), 0, Counter()
    sft_rows_per_q = Counter()
    for row in read_jsonl(args.sft):
        sft_rows += 1
        qid = row["question_id"]
        sft_qids.add(qid)
        sft_rows_per_q[qid] += 1
        provenance = row.get("provenance") or {}
        if isinstance(provenance, str):
            try:
                provenance = json.loads(provenance)
            except json.JSONDecodeError:
                provenance = {}
        index = provenance.get("sample_index")
        if isinstance(index, int):
            sft_selected_indices[index] += 1

    by_question = defaultdict(list)
    for row in read_jsonl(args.rollouts):
        correct = row.get("final_ex")
        if isinstance(correct, str):
            correct = correct == "True"
        by_question[row["question_id"]].append({
            "sample_index": int(row.get("sample_index", 0)),
            "correct": bool(correct),
            "db_id": row.get("db_id"),
        })

    distribution = Counter()
    questions = []
    all_zero, success, first_pass, later_only = [], [], [], []
    per_db = defaultdict(lambda: Counter())
    for qid, rows in by_question.items():
        rows.sort(key=lambda item: item["sample_index"])
        correct_indices = [row["sample_index"] for row in rows if row["correct"]]
        count = len(correct_indices)
        distribution[count] += 1
        db_id = rows[0]["db_id"]
        cls = ("all_zero" if count == 0 else
               "first_pass" if 0 in correct_indices else "later_pass_only")
        in_sft = qid in sft_qids
        record = {
            "question_id": qid, "db_id": db_id, "n4_samples": len(rows),
            "correct": count, "correct_sample_indices": correct_indices,
            "classification": cls, "in_sft": in_sft,
        }
        questions.append(record)
        per_db[db_id]["pool"] += 1
        per_db[db_id]["sft"] += in_sft
        if cls == "all_zero":
            all_zero.append(record)
            per_db[db_id]["all_zero"] += 1
        else:
            success.append(record)
            if cls == "first_pass":
                first_pass.append(record)
                per_db[db_id]["first_pass"] += 1
            else:
                later_only.append(record)
                per_db[db_id]["later_pass_only"] += 1
    questions.sort(key=lambda row: (row["classification"], row["db_id"], row["question_id"]))
    sft_questions = [row for row in questions if row["in_sft"]]
    sft_later = [row for row in sft_questions if row["classification"] == "later_pass_only"]
    sft_first = [row for row in sft_questions if row["classification"] == "first_pass"]
    success_not_sft = [row for row in success if not row["in_sft"]]
    sft_correct_distribution = Counter(row["correct"] for row in sft_questions)
    pool_size = len(questions)
    summary = {
        "pool_questions": pool_size, "n4_rollouts": sum(len(rows) for rows in by_question.values()),
        "n4_samples_per_question": sorted({len(rows) for rows in by_question.values()}),
        "correct_count_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "all_zero": len(all_zero), "all_zero_rate": round(len(all_zero) / pool_size, 4),
        "success": len(success), "success_rate": round(len(success) / pool_size, 4),
        "first_pass": len(first_pass), "later_pass_only": len(later_only),
        "sft_rows": sft_rows, "sft_questions": len(sft_qids),
        "sft_rows_per_question": dict(Counter(sft_rows_per_q.values())),
        "sft_selected_sample_index": dict(sorted(sft_selected_indices.items())),
        "sft_correct_distribution": {str(k): v for k, v in sorted(sft_correct_distribution.items())},
        "sft_first_pass": len(sft_first), "sft_later_pass_only": len(sft_later),
        "sft_all_zero": sum(1 for row in sft_questions if row["classification"] == "all_zero"),
        "success_not_in_sft": len(success_not_sft),
        "success_not_in_sft_ids": [row["question_id"] for row in success_not_sft],
        "sft_not_in_n4": len(sft_qids - set(by_question)),
        "per_db": {db: dict(counts) for db, counts in
                   sorted(per_db.items(), key=lambda item: -item[1]["pool"])},
    }
    for name, rows in (("sft_unanswerable_n4.jsonl", all_zero),
                       ("sft_later_pass_questions.jsonl", sft_later),
                       ("sft_first_pass_questions.jsonl", sft_first),
                       ("sft_success_not_selected.jsonl", success_not_sft)):
        (out / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8")
    (out / "sft_solvability_n4.json").write_text(
        json.dumps({**summary, "questions": questions}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")

    lines = [
        "# SFT training pool: n4 solvability summary", "",
        f"Pool: {pool_size} questions x 4 passes (`p3_full_n4/rollouts.jsonl`).",
        f"SFT file: {sft_rows} rows / {len(sft_qids)} questions "
        "(`sft_student_full_prefer.jsonl`).", "",
        "## Definitions", "",
        "- **first_pass**: sample_index 0 is correct.",
        "- **later_pass_only (multi-pass needed)**: sample 0 is wrong, at least one of 1-3 is correct.",
        "- **all_zero**: 0/4 correct (unanswerable at n4).",
        "", "## Pool (4,190 questions)", "",
        "| correct/4 | questions |", "|---:|---:|",
    ]
    for count in range(5):
        lines.append(f"| {count} | {distribution.get(count, 0)} |")
    lines += [
        "", f"- all_zero: **{len(all_zero)} / {pool_size} = {len(all_zero)/pool_size:.2%}** (unanswerable, excluded from SFT)",
        f"- success: {len(success)} ({len(success)/pool_size:.2%})",
        f"- first_pass: {len(first_pass)}; later_pass_only: {len(later_only)}",
        "", "## SFT training data (2,372 questions)", "",
        f"- rows: {sft_rows}; rows/question: {dict(Counter(sft_rows_per_q.values()))}",
        f"- correct/4 distribution: {dict(sorted(sft_correct_distribution.items()))}",
        f"- first_pass: **{len(sft_first)}** ({len(sft_first)/len(sft_questions):.2%})",
        f"- later_pass_only: **{len(sft_later)}** ({len(sft_later)/len(sft_questions):.2%})",
        f"- all_zero: **{summary['sft_all_zero']}** (SFT is success-filtered; unanswerable questions are absent)",
        f"- success questions not selected into SFT: {len(success_not_sft)} "
        f"({', '.join(summary['success_not_in_sft_ids'])})",
        f"- SFT selected sample_index counts (trajectory-level): {dict(sorted(sft_selected_indices.items()))}",
        "", "## Top databases by all-zero count", "",
        "| db | pool | SFT | all_zero | first_pass | later_pass_only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for db, counts in sorted(per_db.items(), key=lambda item: -item[1]["all_zero"])[:15]:
        lines.append(f"| {db} | {counts['pool']} | {counts['sft']} | {counts['all_zero']} | "
                     f"{counts['first_pass']} | {counts['later_pass_only']} |")
    lines += [
        "", "## Caveats", "",
        "- The n4 collection used `step_600_merged`, E1.1/query/compact, temperature 0.8, top_p 0.95;",
        "  it is not the current step788/E1.2/compact_v2 condition. Solvability is for the",
        "  data-generating model, not a guarantee for the final SFT model.",
        "- The SFT questions and the RL reviewed questions are disjoint (0 intersection);",
        "  this pool is the SFT training pool, not the RL training set.",
        "- The SFT file contains only successful trajectories; it has no negative or",
        "  unanswerable examples. The 1,815 all-zero questions are the excluded set.",
        "", "## Files", "",
        "- `sft_solvability_n4.json` (all 4,190 questions)",
        "- `sft_unanswerable_n4.jsonl` (1,815 all-zero)",
        "- `sft_later_pass_questions.jsonl` (995 SFT questions requiring a later pass)",
        "- `sft_first_pass_questions.jsonl` (1,377 SFT first-pass questions)",
        "- `sft_success_not_selected.jsonl` (3 success questions not in SFT)",
    ]
    (out / "SFT_SOLVABILITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "pool_questions", "all_zero", "all_zero_rate", "success", "first_pass",
        "later_pass_only", "sft_questions", "sft_first_pass", "sft_later_pass_only",
        "sft_all_zero", "success_not_in_sft")}, indent=1))


if __name__ == "__main__":
    main()
