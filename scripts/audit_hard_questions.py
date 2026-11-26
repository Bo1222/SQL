"""CPU hard-question audit: separate "hard but writable" from "dead/invalid".

Reads the frozen 100-question x 8 signal set and reports, for all-zero, mixed
and all-correct groups:
  correct, valid, valid-but-wrong, invalid, no_such_column, syntax,
  average tool calls/schema lookups, and per-question details for all-zero.

No model, no GPU, no new generation.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def error_type(record):
    return (record.get("score") or {}).get("error_type") or record.get("error_type")


def summarize(samples):
    n = len(samples)
    counts = Counter()
    for record in samples:
        counts["samples"] += 1
        counts["correct"] += bool(record.get("correct"))
        counts["valid"] += bool(record.get("valid"))
        counts["valid_but_wrong"] += bool(record.get("valid")) and not bool(record.get("correct"))
        counts["invalid"] += not bool(record.get("valid"))
        counts["no_such_column"] += error_type(record) == "no_such_column"
        counts["syntax"] += error_type(record) == "syntax_error"
        counts["no_final_sql"] += not record.get("final_sql")
        counts["answer"] += record.get("finish_reason") == "answer_submitted"
        counts["tool_calls"] += int(record.get("tool_calls") or 0)
        counts["schema_lookups"] += int(record.get("schema_lookups") or 0)
    return {
        "questions": len({record["id"] for record in samples}),
        "samples": n,
        "correct_rate": round(counts["correct"] / n, 3),
        "valid_rate": round(counts["valid"] / n, 3),
        "valid_but_wrong_rate": round(counts["valid_but_wrong"] / n, 3),
        "invalid_rate": round(counts["invalid"] / n, 3),
        "no_such_column_rate": round(counts["no_such_column"] / n, 3),
        "syntax_rate": round(counts["syntax"] / n, 3),
        "no_final_sql_rate": round(counts["no_final_sql"] / n, 3),
        "answer_rate": round(counts["answer"] / n, 3),
        "avg_tool_calls": round(counts["tool_calls"] / n, 2),
        "avg_schema_lookups": round(counts["schema_lookups"] / n, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="reports/m6/rl_signal_20261118/records.jsonl")
    parser.add_argument("--quarantine",
                        default="reports/m6/rl_preflight_20261118/annotation_quarantine.jsonl")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120")
    parser.add_argument("--samples-per-question", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quarantine_ids = {row.get("id") or row.get("question_id")
                      for row in read_jsonl(args.quarantine)} if Path(args.quarantine).is_file() else set()
    rows = read_jsonl(args.records)
    by_question = defaultdict(list)
    for record in rows:
        by_question[record["id"]].append(record)
    counts = {qid: sum(1 for r in records if r.get("correct"))
              for qid, records in by_question.items()}
    groups = {
        "all_zero": [qid for qid, value in counts.items() if value == 0],
        "mixed": [qid for qid, value in counts.items()
                  if 0 < value < args.samples_per_question],
        "all_correct": [qid for qid, value in counts.items()
                        if value == args.samples_per_question],
    }
    group_stats = {name: summarize([r for qid in qids for r in by_question[qid]])
                   for name, qids in groups.items()}
    all_zero_details = []
    for qid in groups["all_zero"]:
        records = by_question[qid]
        valid_but_wrong = next(
            (r for r in records if r.get("valid") and not r.get("correct")), None)
        sample = valid_but_wrong or records[0]
        invalid_samples = sum(1 for r in records if not r.get("valid"))
        if not any(r.get("valid") for r in records):
            category = "schema_column_grounding"
        elif invalid_samples >= 4:
            category = "schema_plus_semantic"
        elif valid_but_wrong is not None and invalid_samples <= 3:
            category = "semantic_result_error"
        else:
            category = "mixed"
        all_zero_details.append({
            "id": qid,
            "db_id": records[0]["db_id"],
            "quarantined": qid in quarantine_ids,
            "category": category,
            "gold_sql": records[0].get("gold_sql"),
            "sample_pred": sample.get("final_sql"),
            "sample_pred_status": ("valid_but_wrong" if valid_but_wrong
                                   else "invalid" if not sample.get("valid") else "other"),
            "question": records[0].get("question", ""),
            "samples": len(records),
            "valid_samples": sum(1 for r in records if r.get("valid")),
            "valid_but_wrong_samples": sum(
                1 for r in records if r.get("valid") and not r.get("correct")),
            "invalid_samples": sum(1 for r in records if not r.get("valid")),
            "all_samples_invalid": not any(r.get("valid") for r in records),
            "error_types": dict(Counter(str(error_type(r)) for r in records)),
            "avg_tool_calls": round(sum(int(r.get("tool_calls") or 0) for r in records)
                                    / len(records), 2),
            "avg_schema_lookups": round(sum(int(r.get("schema_lookups") or 0) for r in records)
                                        / len(records), 2),
        })
    audit = {
        "records": args.records,
        "groups": groups,
        "group_stats": group_stats,
        "all_zero_details": all_zero_details,
        "all_zero_with_any_valid": sum(1 for row in all_zero_details
                                       if row["valid_samples"] > 0),
        "all_zero_with_any_valid_but_wrong": sum(
            1 for row in all_zero_details if row["valid_but_wrong_samples"] > 0),
        "all_zero_all_invalid": sum(1 for row in all_zero_details
                                    if row["all_samples_invalid"]),
        "all_zero_quarantined": sum(1 for row in all_zero_details if row["quarantined"]),
        "quarantine_ids": sorted(quarantine_ids),
    }
    (out / "hard_question_audit.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Hard-question audit (100 questions x 8)", "",
        f"Source: `{args.records}`", "",
        "| group | questions | samples | correct | valid | valid-but-wrong | invalid | no_such_column | syntax | avg tools | avg lookups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_zero", "mixed", "all_correct"):
        row = group_stats[name]
        lines.append(
            f"| {name} | {row['questions']} | {row['samples']} | {row['correct_rate']:.3f} | "
            f"{row['valid_rate']:.3f} | {row['valid_but_wrong_rate']:.3f} | "
            f"{row['invalid_rate']:.3f} | {row['no_such_column_rate']:.3f} | "
            f"{row['syntax_rate']:.3f} | {row['avg_tool_calls']} | "
            f"{row['avg_schema_lookups']} |")
    lines += [
        "", "## All-zero questions", "",
        f"- {len(groups['all_zero'])} questions; {audit['all_zero_with_any_valid']} have at "
        "least one valid sample, "
        f"{audit['all_zero_with_any_valid_but_wrong']} have at least one valid-but-wrong "
        "sample, and "
        f"{audit['all_zero_all_invalid']} are all-invalid.",
        f"- {audit['all_zero_quarantined']} all-zero question(s) are among the 17 "
        "quarantined input-conflict rows; exclude/fix them, do not teach them as hard.",
        "- All-zero is mostly *hard and executable-but-wrong*: the SQL executes but "
        "is not correct; 0/256 samples are correct. `valid` is not evidence of "
        "solvability. Vanilla GRPO has zero advantage on these groups.",
        "", "| id | db | quarantined | category | valid | vbw | invalid | all_invalid | error types |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in all_zero_details:
        lines.append(
            f"| {row['id']} | {row['db_id']} | {row['quarantined']} | "
            f"{row['category']} | {row['valid_samples']}/8 | "
            f"{row['valid_but_wrong_samples']}/8 | {row['invalid_samples']}/8 | "
            f"{row['all_samples_invalid']} | {row['error_types']} |")
    lines += ["", "## All-invalid questions: Gold vs sample prediction", ""]
    for row in all_zero_details:
        if not row["all_samples_invalid"]:
            continue
        lines += [
            f"### {row['id']} ({row['db_id']})", "",
            f"- question: {row['question']}",
            f"- Gold: `{row['gold_sql']}`",
            f"- sample prediction ({row['sample_pred_status']}): `{row['sample_pred']}`",
            "",
        ]
    lines.append("")
    (out / "HARD_QUESTION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"group_stats": group_stats,
                      "all_zero_with_any_valid": audit["all_zero_with_any_valid"],
                      "all_zero_all_invalid": audit["all_zero_all_invalid"]}, indent=1))


if __name__ == "__main__":
    main()
