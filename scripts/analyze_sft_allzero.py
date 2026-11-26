"""Root-cause audit of the SFT pool questions that were 0/4 at n4.

For each of the 1,815 all-zero questions:
  - aggregate the 4 rollout errors (valid, valid-but-wrong, invalid, no SQL,
    infrastructure, error types)
  - execute the Gold SQL read-only (one cached connection per DB)
  - classify the dominant failure: Gold/execution defect, empty Gold,
    schema/column grounding, semantic valid-but-wrong, or format/tool
  - compare feature statistics with a sample of solved questions

No model, no GPU, no API.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.sql_dependencies import sqlite_requirements  # noqa: E402


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    return value or {}


class GoldAuditor:
    def __init__(self, timeout_s=4.0):
        self.timeout_s = timeout_s
        self.connections = {}

    def connection(self, db_path):
        if db_path not in self.connections:
            con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1",
                                  uri=True)
            con.execute("PRAGMA query_only=ON")
            self.connections[db_path] = con
        return self.connections[db_path]

    def run(self, db_path, sql):
        out = {"gold_ok": False, "gold_rows": None, "gold_empty": None,
               "gold_error": None, "n_joins": len(re.findall(r"\bJOIN\b", sql, flags=re.I))}
        try:
            con = self.connection(db_path)
            deadline = time.monotonic() + self.timeout_s
            con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
            try:
                rows = con.execute(sql).fetchmany(1001)
            finally:
                con.set_progress_handler(None, 0)
            out.update({"gold_ok": True, "gold_rows": len(rows),
                        "gold_empty": len(rows) == 0})
        except Exception as exc:  # noqa: BLE001
            out["gold_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return out

    def requirements(self, db_path, sql):
        try:
            req = sqlite_requirements(db_path, sql)
            return {"requirements_ok": req.get("ok"),
                    "requirements_error": req.get("error"),
                    "tables": len(req.get("tables", [])),
                    "columns": len(req.get("columns", []))}
        except Exception as exc:  # noqa: BLE001
            return {"requirements_ok": False,
                    "requirements_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "tables": None, "columns": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts",
                        default="reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl")
    parser.add_argument("--allzero",
                        default="reports/m6/sft_first_20261120/sft_unanswerable_n4.jsonl")
    parser.add_argument("--solvability",
                        default="reports/m6/sft_first_20261120/sft_solvability_n4.json")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120")
    parser.add_argument("--solved-sample", type=int, default=300)
    parser.add_argument("--requirements-sample", type=int, default=300)
    parser.add_argument("--gold-timeout", type=float, default=4.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allzero = {row["question_id"]: row for row in read_jsonl(args.allzero)}
    solvability = json.loads(Path(args.solvability).read_text())
    solved_ids = {row["question_id"] for row in solvability["questions"]
                  if row["classification"] != "all_zero"}
    by_question = defaultdict(list)
    meta = {}
    for row in read_jsonl(args.rollouts):
        qid = row["question_id"]
        if qid not in allzero and qid not in solved_ids:
            continue
        score = parse(row.get("score"))
        record = parse(row.get("record"))
        by_question[qid].append({
            "correct": bool(score.get("correct")),
            "valid": bool(score.get("ok")),
            "error_type": score.get("error_type"),
            "final_sql": row.get("final_sql"),
            "infrastructure_error": bool(score.get("infrastructure_error"))
            or bool(record.get("infrastructure_error")),
            "finish_reason": record.get("finish_reason"),
        })
        if qid not in meta:
            meta[qid] = {
                "db_path": record.get("db_path"),
                "gold_sql": record.get("gold_sql"),
                "question": record.get("question", ""),
                "evidence": record.get("evidence", "") or "",
                "db_id": row.get("db_id"),
            }
    auditor = GoldAuditor(timeout_s=args.gold_timeout)
    audit_rows = []
    for qid in sorted(allzero):
        samples = by_question.get(qid, [])
        valid = sum(row["valid"] for row in samples)
        valid_wrong = sum(row["valid"] and not row["correct"] for row in samples)
        invalid = sum(not row["valid"] for row in samples)
        no_sql = sum(not row["final_sql"] for row in samples)
        infra = sum(row["infrastructure_error"] for row in samples)
        error_types = Counter(str(row["error_type"]) for row in samples)
        gold = auditor.run(meta[qid]["db_path"], meta[qid]["gold_sql"])
        if gold["gold_error"]:
            category = "gold_execution_defect"
        elif gold["gold_empty"]:
            category = "empty_gold_result"
        elif valid == 0:
            if error_types.get("no_such_column", 0) + error_types.get("no_such_table", 0) >= 2:
                category = "schema_column_grounding"
            elif no_sql or error_types.get("invalid_action", 0) or error_types.get("syntax_error", 0):
                category = "format_or_tool"
            else:
                category = "invalid_other"
        elif valid_wrong > 0:
            category = "semantic_valid_but_wrong"
        else:
            category = "mixed"
        audit_rows.append({
            "question_id": qid, "db_id": meta[qid]["db_id"],
            "category": category,
            "samples": len(samples), "valid": valid,
            "valid_but_wrong": valid_wrong, "invalid": invalid,
            "no_sql": no_sql, "infrastructure": infra,
            "error_types": dict(error_types), "gold": gold,
            "question_chars": len(meta[qid]["question"]),
            "evidence_chars": len(meta[qid]["evidence"]),
        })
    rng = random.Random(0)
    requirements_sample = (rng.sample(sorted(allzero), min(args.requirements_sample, len(allzero)))
                           if allzero else [])
    for row in audit_rows:
        if row["question_id"] in requirements_sample:
            row["gold"].update(auditor.requirements(meta[row["question_id"]]["db_path"],
                                                    meta[row["question_id"]]["gold_sql"]))
    solved_sample = (rng.sample(sorted(solved_ids), min(args.solved_sample, len(solved_ids)))
                     if solved_ids else [])
    solved_features = []
    for qid in solved_sample:
        if qid not in meta:
            continue
        gold = auditor.run(meta[qid]["db_path"], meta[qid]["gold_sql"])
        gold.update(auditor.requirements(meta[qid]["db_path"], meta[qid]["gold_sql"]))
        solved_features.append(gold)
    def avg(values):
        return round(sum(values) / len(values), 2) if values else None
    categories = Counter(row["category"] for row in audit_rows)
    by_db = Counter(row["db_id"] for row in audit_rows)
    summary = {
        "all_zero_questions": len(audit_rows),
        "categories": dict(categories.most_common()),
        "category_rate": {key: round(value / len(audit_rows), 4)
                          for key, value in categories.items()},
        "top_dbs": dict(by_db.most_common(20)),
        "mean_features_all_zero": {
            "question_chars": avg([row["question_chars"] for row in audit_rows]),
            "evidence_chars": avg([row["evidence_chars"] for row in audit_rows]),
            "gold_joins": avg([row["gold"]["n_joins"] for row in audit_rows]),
            "gold_rows": avg([row["gold"]["gold_rows"] for row in audit_rows
                              if row["gold"]["gold_rows"] is not None]),
        },
        "requirements_sample": len(requirements_sample),
        "mean_features_requirements_sample": {
            "gold_tables": avg([row["gold"]["tables"] for row in audit_rows
                                if row["gold"].get("tables") is not None]),
            "gold_columns": avg([row["gold"]["columns"] for row in audit_rows
                                 if row["gold"].get("columns") is not None]),
        },
        "solved_sample": len(solved_features),
        "solved_control_mean_features": {
            "gold_tables": avg([row["tables"] for row in solved_features
                                if row["tables"] is not None]),
            "gold_columns": avg([row["columns"] for row in solved_features
                                 if row["columns"] is not None]),
            "gold_joins": avg([row["n_joins"] for row in solved_features]),
            "gold_rows": avg([row["gold_rows"] for row in solved_features
                              if row["gold_rows"] is not None]),
        },
        "gold_errors": sum(1 for row in audit_rows if row["gold"]["gold_error"]),
        "empty_gold": sum(1 for row in audit_rows if row["gold"]["gold_empty"]),
    }
    (out / "sft_allzero_audit.json").write_text(
        json.dumps({"summary": summary, "questions": audit_rows}, indent=1,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# SFT 0/4 questions: root-cause audit", "",
        f"Questions: {len(audit_rows)} (all 0/4 in the n4 SFT pool).", "",
        "## Categories", "", "| category | questions | rate |", "|---|---:|---:|",
    ]
    for key, value in categories.most_common():
        lines.append(f"| {key} | {value} | {value/len(audit_rows):.2%} |")
    lines += [
        "", "## Gold audit", "",
        f"- Gold execution errors: {summary['gold_errors']}",
        f"- Empty Gold result: {summary['empty_gold']}",
        f"- requirements/Gold feature sample: {summary['requirements_sample']}",
        "", "## Mean features (0/4 vs solved control)", "",
        f"- question chars: {summary['mean_features_all_zero']['question_chars']}",
        f"- evidence chars: {summary['mean_features_all_zero']['evidence_chars']}",
        f"- Gold joins: {summary['mean_features_all_zero']['gold_joins']} vs "
        f"{summary['solved_control_mean_features']['gold_joins']}",
        f"- Gold rows: {summary['mean_features_all_zero']['gold_rows']} vs "
        f"{summary['solved_control_mean_features']['gold_rows']}",
        f"- Gold tables (sample): {summary['mean_features_requirements_sample']['gold_tables']} vs "
        f"{summary['solved_control_mean_features']['gold_tables']}",
        f"- Gold columns (sample): {summary['mean_features_requirements_sample']['gold_columns']} vs "
        f"{summary['solved_control_mean_features']['gold_columns']}",
        "", "## Top databases", "", "| db | questions |", "|---|---:|",
    ]
    for db, value in by_db.most_common(15):
        lines.append(f"| {db} | {value} |")
    lines += ["", "## Examples", ""]
    for category in [key for key, _ in categories.most_common()]:
        examples = [row for row in audit_rows if row["category"] == category][:3]
        lines += [f"### {category}", ""]
        for row in examples:
            lines.append(
                f"- `{row['question_id']}` ({row['db_id']}): valid={row['valid']}/4, "
                f"valid-but-wrong={row['valid_but_wrong']}/4, invalid={row['invalid']}/4, "
                f"errors={row['error_types']}, Gold rows={row['gold']['gold_rows']}, "
                f"Gold error={row['gold']['gold_error']}")
        lines.append("")
    (out / "SFT_ALLZERO_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
