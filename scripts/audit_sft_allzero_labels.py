"""Step 0 CPU triage for the 1,815 n4-all-zero SFT-pool questions.

For every question this script:
  * loads question / Evidence / Gold / db_path from the rollout record;
  * executes Gold read-only (4 s, then one 30 s retry for slow queries);
  * checks simple, reproducible consistency signals between question, Evidence,
    Gold, and the live SQLite schema;
  * writes ``dead / evidence_conflict / ambiguous / label_clean`` triage labels.

These labels are CPU heuristics, not a human semantic audit. In particular,
``label_clean`` means "no CPU-detectable flag", not "verified correct". The
output is for selecting what to audit/fix next, never as an SFT target.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sft_hard_question_common import (  # noqa: E402
    SchemaCatalog,
    content_tokens,
    execute_gold,
    extract_evidence_filters,
    gold_requirements,
    is_format_only,
    normalize_text,
    read_jsonl,
    resolve_filter_column,
    sql_string_literals,
    text_quoted_literals,
    write_jsonl,
)


PERCENT_RE = re.compile(r"\b(?:percentage|percent|ratio|proportion)\b", re.I)
AVERAGE_RE = re.compile(r"\b(?:average|mean)\b", re.I)
HOW_MANY_RE = re.compile(r"\b(?:how many|count of)\b", re.I)
TOTAL_RE = re.compile(r"\b(?:total|overall)\b", re.I)
SUPERLATIVE_RE = re.compile(
    r"\b(?:most|least|highest|lowest|greatest|smallest|top|bottom|"
    r"maximum|minimum|latest|earliest|oldest|youngest)\b", re.I)
DISTINCT_RE = re.compile(r"\b(?:distinct|unique)\b", re.I)
NEGATION_RE = re.compile(r"\b(?:not|excluding|without|never)\b", re.I)
EVIDENCE_REF_RE = re.compile(
    r"(?<![\w.])([A-Za-z_][\w\-]*)\.([A-Za-z_][\w\-]*)(?![\w.])")


def add(container: list[dict], code: str, detail: str | None = None, **extra):
    item = {"code": code}
    if detail:
        item["detail"] = detail
    item.update(extra)
    container.append(item)


def gold_has_any(gold_lower: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in gold_lower for pattern in patterns)


def check_evidence_refs(evidence: str, catalog: SchemaCatalog,
                        conflicts: list[dict], ambiguities: list[dict]):
    if not evidence:
        return
    seen = set()
    for match in EVIDENCE_REF_RE.finditer(evidence):
        table_raw, column = match.groups()
        if re.fullmatch(r"[Tt]\d+", table_raw):
            continue
        if len(table_raw) == 1 or len(column) == 1:
            continue
        key = (table_raw.lower(), column.lower())
        if key in seen:
            continue
        seen.add(key)
        info = catalog.table(table_raw)
        if info:
            if not catalog.has_column(info["name"], column):
                add(conflicts, "evidence_column_not_in_schema",
                    f"{info['name']}.{column} not in live schema")
        elif catalog.column_tables(table_raw) and catalog.column_tables(column):
            # Path notation such as ``w1st.word``: both sides are columns, not
            # a table.column reference. It is not a label conflict.
            continue
        elif catalog.column_tables(column):
            add(ambiguities, "evidence_table_unknown",
                f"{table_raw}.{column}: table unknown but column exists")
        else:
            add(ambiguities, "evidence_unresolved_ref",
                f"{table_raw}.{column} unresolved")


def check_evidence_literals(evidence: str, question: str, gold: str,
                            catalog: SchemaCatalog, ambiguities: list[dict]):
    if not evidence:
        return
    gold_norm = normalize_text(gold)
    question_norm = normalize_text(question)
    schema_names = {normalize_text(name) for name in catalog.tables}
    schema_names.update(
        normalize_text(col["name"])
        for info in catalog.tables.values() for col in info["columns"])
    for value in text_quoted_literals(evidence):
        if len(value) < 3 or is_format_only(value):
            continue
        value_norm = normalize_text(value)
        if not value_norm:
            continue
        # Evidence often quotes a column/table name ("Product Name"); that is
        # not an unused data value.
        if value_norm in schema_names:
            continue
        if value_norm not in gold_norm and value_norm not in question_norm:
            add(ambiguities, "evidence_literal_unused", value)


def check_evidence_filters(evidence: str, gold: str, catalog: SchemaCatalog,
                           conflicts: list[dict], ambiguities: list[dict]):
    gold_lower = (gold or "").lower()
    gold_norm = normalize_text(gold)
    for item in extract_evidence_filters(evidence):
        table, column = resolve_filter_column(catalog, item)
        raw_table = item.get("table_hint")
        raw_column = item.get("column", "")
        if table is None and raw_table:
            if not catalog.table(raw_table):
                if catalog.column_tables(raw_table):
                    # Paired-column notation such as ``w2nd.word`` rather than
                    # a real table name. Skip; the Gold SQL is the join path.
                    continue
                add(conflicts, "evidence_filter_unknown_table", raw_table)
                continue
            if column is None:
                add(conflicts, "evidence_filter_column_not_in_schema",
                    f"{raw_table}.{raw_column}")
                continue
        if table is None and not raw_table:
            candidates = catalog.column_tables(raw_column)
            fuzzy = catalog.fuzzy_columns(raw_column) if not candidates else []
            if not candidates and not fuzzy:
                add(conflicts, "evidence_filter_column_not_in_schema", raw_column)
                continue
            if len(fuzzy) == 1:
                table, column = fuzzy[0]
            elif len(candidates) == 1:
                info = catalog.tables[candidates[0]]
                column = next(col["name"] for col in info["columns"]
                              if col["name"].lower() == raw_column.lower())
                table = info["name"]
            elif len(candidates) > 1 or len(fuzzy) > 1:
                # The column name is shared across tables. If Gold contains
                # both the column and value, the ambiguity is harmless; if not,
                # Evidence and Gold disagree.
                value = str(item.get("value", ""))
                value_norm = normalize_text(value)
                if raw_column.lower() in gold_lower:
                    if value_norm and value_norm not in gold_norm:
                        add(conflicts, "evidence_filter_value_not_in_gold",
                            f"{raw_column} {item.get('op')} {value}")
                else:
                    add(conflicts, "evidence_filter_column_not_in_gold",
                        raw_column)
                continue
            else:
                add(conflicts, "evidence_filter_column_not_in_schema", raw_column)
                continue
        if column is None:
            add(conflicts, "evidence_filter_column_not_in_schema", raw_column)
            continue
        if column.lower() not in gold_lower:
            add(conflicts, "evidence_filter_column_not_in_gold",
                f"{table}.{column}")
        value = str(item.get("value", ""))
        value_norm = normalize_text(value)
        if value_norm and value_norm not in gold_norm:
            add(conflicts, "evidence_filter_value_not_in_gold",
                f"{table}.{column} {item.get('op')} {value}")


def check_question(question: str, gold: str, ambiguities: list[dict]):
    question = question or ""
    gold_lower = (gold or "").lower()
    gold_norm = normalize_text(gold)
    for value in text_quoted_literals(question):
        if len(value) < 2 or is_format_only(value):
            continue
        value_norm = normalize_text(value)
        if value_norm and value_norm not in gold_norm:
            add(ambiguities, "question_literal_missing_from_gold", value)

    question_lower = question.lower()
    if PERCENT_RE.search(question):
        if not (gold_has_any(gold_lower, ("* 100", "*100", "/"))
                or ("cast(" in gold_lower and "/" in gold_lower)):
            add(ambiguities, "aggregate_mismatch", "question asks percentage")
    elif AVERAGE_RE.search(question):
        average_ok = (
            gold_has_any(gold_lower, ("avg(", "average("))
            or (gold_has_any(gold_lower, ("/ count(", "/count(")))
            or (gold_has_any(gold_lower, ("sum(",)) and "count(" in gold_lower
                and "/" in gold_lower)
        )
        if not average_ok:
            add(ambiguities, "aggregate_mismatch", "question asks average")
    elif HOW_MANY_RE.search(question):
        # SUM(CASE ... THEN 1 ELSE 0) is also a count. A superlative "how many
        # ... in the country with the highest number" is answered by MAX(value).
        max_answer = bool(re.search(
            r"\b(?:highest|most|largest|greatest|maximum)\b", question, re.I))
        if not gold_has_any(gold_lower, ("count(", "sum(")) and not (
                max_answer and "max(" in gold_lower):
            add(ambiguities, "aggregate_mismatch", "question asks count")
    elif TOTAL_RE.search(question):
        if not gold_has_any(gold_lower, ("sum(", "count(")):
            add(ambiguities, "aggregate_mismatch", "question asks total")

    for match in SUPERLATIVE_RE.finditer(question):
        prefix = question[max(0, match.start() - 3):match.start()].lower()
        if prefix.endswith("at "):
            continue
        if not (gold_has_any(gold_lower, ("order by",))
                or re.search(r"\b(?:max|min)\s*\(", gold_lower)):
            add(ambiguities, "order_mismatch", "superlative without ORDER BY/MAX/MIN")
            break

    if DISTINCT_RE.search(question):
        if not (gold_has_any(gold_lower, ("distinct", "group by"))):
            add(ambiguities, "distinct_mismatch", "distinct/unique without DISTINCT/GROUP BY")

    if NEGATION_RE.search(question):
        if not (gold_has_any(gold_lower, (" not ", "not in", "except", "left join",
                                          "is null", "<>", "!="))):
            add(ambiguities, "negation_mismatch", "negation without NOT/EXCEPT/LEFT JOIN")


def check_gold_literals(question: str, evidence: str, gold: str,
                        ambiguities: list[dict]):
    context = normalize_text(f"{question or ''} {evidence or ''}")
    unmentioned = []
    for value in sql_string_literals(gold):
        if len(value) < 2 or is_format_only(value):
            continue
        value_norm = normalize_text(value)
        if value_norm and value_norm not in context:
            unmentioned.append(value)
    if unmentioned:
        add(ambiguities, "gold_literal_unmentioned",
            ", ".join(unmentioned[:5]), count=len(unmentioned))


def semantic_subclass(row: dict) -> str | None:
    if row.get("category") != "semantic_valid_but_wrong":
        return None
    valid = int(row.get("valid", 0))
    if valid >= 4:
        return "semantic_all4_valid"
    if valid == 3:
        return "semantic_3of4_valid"
    if valid == 2:
        return "semantic_2of4_valid"
    return "semantic_1of4_valid"


def schema_subclass(row: dict) -> str | None:
    if row.get("category") != "schema_column_grounding":
        return None
    errors = row.get("error_types") or {}
    column = int(errors.get("no_such_column", 0))
    table = int(errors.get("no_such_table", 0))
    syntax = int(errors.get("syntax_error", 0))
    no_sql = int(row.get("no_sql", 0))
    if column and table and syntax:
        return "schema_column_table_syntax"
    if column and table:
        return "schema_column_table"
    if column and syntax:
        return "schema_column_syntax"
    if column and no_sql:
        return "schema_column_plus_no_sql"
    if column:
        return "schema_column_only"
    if table and no_sql:
        return "schema_table_plus_no_sql"
    if table:
        return "schema_table_only"
    if no_sql:
        return "schema_no_sql"
    return "schema_other"


def classify(gold: dict, dead: list[dict], conflicts: list[dict],
             ambiguities: list[dict]) -> str:
    if dead:
        return "dead"
    if conflicts:
        return "evidence_conflict"
    if ambiguities:
        return "ambiguous"
    return "label_clean"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts",
                        default="reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl")
    parser.add_argument("--allzero",
                        default="reports/m6/sft_first_20261120/sft_unanswerable_n4.jsonl")
    parser.add_argument("--audit",
                        default="reports/m6/sft_first_20261120/sft_allzero_audit.json")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120/step0_step1")
    parser.add_argument("--gold-timeout", type=float, default=4.0)
    parser.add_argument("--retry-timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allzero = {row["question_id"]: row for row in read_jsonl(args.allzero)}
    audit = {row["question_id"]: row for row in
             json.loads(Path(args.audit).read_text())["questions"]}
    ids = sorted(allzero)
    if args.limit:
        ids = ids[:args.limit]
    wanted = set(ids)

    meta: dict[str, dict] = {}
    for row in read_jsonl(args.rollouts):
        qid = row["question_id"]
        if qid in wanted and qid not in meta:
            meta[qid] = row.get("record") or {}

    catalogs: dict[str, SchemaCatalog | Exception] = {}
    rows = []
    for index, qid in enumerate(ids, 1):
        audit_row = audit.get(qid, {})
        base = allzero[qid]
        record = meta.get(qid) or {}
        question = record.get("question", "") or ""
        evidence = record.get("evidence", "") or ""
        gold_sql = record.get("gold_sql", "") or ""
        db_path = record.get("db_path", "") or ""
        dead: list[dict] = []
        conflicts: list[dict] = []
        ambiguities: list[dict] = []
        gold_info = {}
        requirements = {}

        if not record:
            add(dead, "missing_rollout_record")
        elif not Path(db_path).exists():
            add(dead, "missing_database", db_path)
        else:
            if db_path not in catalogs:
                try:
                    catalogs[db_path] = SchemaCatalog(db_path)
                except Exception as exc:  # noqa: BLE001
                    catalogs[db_path] = exc
            catalog = catalogs[db_path]
            if isinstance(catalog, Exception):
                add(dead, "schema_unreadable", str(catalog))
            else:
                initial = execute_gold(db_path, gold_sql, args.gold_timeout)
                final = initial
                if (not initial["ok"] or initial["timeout"]) and args.retry_timeout > args.gold_timeout:
                    final = execute_gold(db_path, gold_sql, args.retry_timeout)
                gold_info = {"initial": initial, "final": final}
                if not final["ok"]:
                    add(dead, "gold_execution_error", final.get("error") or "unknown")
                else:
                    if initial["timeout"]:
                        add(ambiguities, "slow_gold_query",
                            f"needed {final['elapsed_s']}s (initial {initial['elapsed_s']}s)")
                    if final["empty"]:
                        add(ambiguities, "empty_gold_result")
                    try:
                        requirements = gold_requirements(db_path, gold_sql)
                    except Exception as exc:  # noqa: BLE001
                        requirements = {"ok": False, "error": str(exc),
                                        "tables": [], "columns": []}
                    if not requirements.get("ok"):
                        add(ambiguities, "gold_requirements_unavailable",
                            requirements.get("error"))
                    check_evidence_refs(evidence, catalog, conflicts, ambiguities)
                    check_evidence_filters(evidence, gold_sql, catalog,
                                           conflicts, ambiguities)
                    check_question(question, gold_sql, ambiguities)
                    check_gold_literals(question, evidence, gold_sql, ambiguities)
                    check_evidence_literals(evidence, question, gold_sql,
                                            catalog, ambiguities)

        status = classify(gold_info.get("final", {}), dead, conflicts, ambiguities)
        rows.append({
            "question_id": qid,
            "db_id": record.get("db_id") or base.get("db_id"),
            "category": audit_row.get("category", "unknown"),
            "semantic_subclass": semantic_subclass(audit_row),
            "schema_subclass": schema_subclass(audit_row),
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "db_path": db_path,
            "gold": gold_info,
            "gold_requirements": {
                "ok": requirements.get("ok"),
                "error": requirements.get("error"),
                "tables": requirements.get("tables", []),
                "columns": requirements.get("columns", []),
            },
            "label": {
                "status": status,
                "dead_reasons": dead,
                "evidence_conflicts": conflicts,
                "ambiguities": ambiguities,
                "not_for_training": True,
            },
        })
        if index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {qid} {status}", flush=True)

    write_jsonl(out / "label_audit.jsonl", rows)
    write_jsonl(out / "sft_allzero_subclasses.jsonl", [{
        "question_id": row["question_id"],
        "db_id": row["db_id"],
        "category": row["category"],
        "semantic_subclass": row["semantic_subclass"],
        "schema_subclass": row["schema_subclass"],
        "valid": audit.get(row["question_id"], {}).get("valid"),
        "valid_but_wrong": audit.get(row["question_id"], {}).get("valid_but_wrong"),
        "invalid": audit.get(row["question_id"], {}).get("invalid"),
        "no_sql": audit.get(row["question_id"], {}).get("no_sql"),
        "error_types": audit.get(row["question_id"], {}).get("error_types"),
        "label_status": row["label"]["status"],
        "evidence_conflict_codes": [x["code"] for x in row["label"]["evidence_conflicts"]],
        "ambiguity_codes": [x["code"] for x in row["label"]["ambiguities"]],
    } for row in rows])

    status_counts = Counter(row["label"]["status"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    status_by_category = defaultdict(Counter)
    status_by_subclass = defaultdict(Counter)
    for row in rows:
        status_by_category[row["category"]][row["label"]["status"]] += 1
        subclass = row["semantic_subclass"] or row["schema_subclass"]
        if subclass:
            status_by_subclass[subclass][row["label"]["status"]] += 1
    conflict_codes = Counter(item["code"] for row in rows
                             for item in row["label"]["evidence_conflicts"])
    ambiguity_codes = Counter(item["code"] for row in rows
                              for item in row["label"]["ambiguities"])
    subclass_counts = Counter(
        row["semantic_subclass"] or row["schema_subclass"]
        for row in rows if row["semantic_subclass"] or row["schema_subclass"])
    slow_gold = sum(1 for row in rows
                    if any(x["code"] == "slow_gold_query"
                           for x in row["label"]["ambiguities"]))
    dead_rows = [row for row in rows if row["label"]["status"] == "dead"]
    summary = {
        "questions": len(rows),
        "status_counts": dict(status_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "status_by_category": {key: dict(value.most_common())
                               for key, value in status_by_category.items()},
        "status_by_subclass": {key: dict(value.most_common())
                               for key, value in status_by_subclass.items()},
        "subclass_counts": dict(subclass_counts.most_common()),
        "evidence_conflict_codes": dict(conflict_codes.most_common()),
        "ambiguity_codes": dict(ambiguity_codes.most_common()),
        "slow_gold_questions": slow_gold,
        "dead_examples": dead_rows[:5],
        "gold_timeout_initial_s": args.gold_timeout,
        "gold_timeout_retry_s": args.retry_timeout,
        "caveat": "CPU heuristic triage; label_clean is not a semantic verification.",
        "not_for_training": True,
    }
    (out / "label_audit_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    by_status = defaultdict(list)
    for row in rows:
        by_status[row["label"]["status"]].append(row)
    for status, status_rows in by_status.items():
        write_jsonl(out / f"labels_{status}.jsonl", status_rows)

    lines = [
        "# Step 0: CPU label/Evidence triage", "",
        f"Questions: {len(rows)}. Gold timeout: {args.gold_timeout}s, "
        f"retry: {args.retry_timeout}s. No model, GPU, or API.",
        "", "## Status", "", "| status | questions | rate |", "|---|---:|---:|",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"| {status} | {count} | {count/len(rows):.2%} |")
    lines += [
        "", "## Status by original category", "",
        "| category | total | " + " | ".join(status_counts) + " |",
        "|---|---:|" + "---:|" * len(status_counts),
    ]
    for category, count in category_counts.most_common():
        values = [str(status_by_category[category].get(status, 0))
                  for status in status_counts]
        lines.append(f"| {category} | {count} | " + " | ".join(values) + " |")
    lines += [
        "", "## Subclasses", "", "| subclass | questions |", "|---|---:|",
    ]
    for name, count in subclass_counts.most_common():
        lines.append(f"| {name} | {count} |")
    lines += [
        "", "## Status by subclass", "",
        "| subclass | total | " + " | ".join(status_counts) + " |",
        "|---|---:|" + "---:|" * len(status_counts),
    ]
    for name, count in subclass_counts.most_common():
        values = [str(status_by_subclass[name].get(status, 0))
                  for status in status_counts]
        lines.append(f"| {name} | {count} | " + " | ".join(values) + " |")
    lines += [
        "", f"Slow Gold (>={args.gold_timeout:g}s, retried): {slow_gold}",
        f"Dead: {len(dead_rows)}",
        "", "## Evidence-conflict codes", "",
        "| code | questions |", "|---|---:|",
    ]
    for code, count in conflict_codes.most_common(20):
        lines.append(f"| {code} | {count} |")
    lines += ["", "## Ambiguity codes", "", "| code | questions |", "|---|---:|"]
    for code, count in ambiguity_codes.most_common(25):
        lines.append(f"| {code} | {count} |")
    lines += [
        "", "## Caveat", "",
        "`label_clean` means no CPU-detectable flag, not verified semantic",
        "correctness. `evidence_conflict` means the Evidence and Gold disagree on",
        "a simple column/value check; it is a review queue, not a final label.",
        "The output is not a training file.", "",
    ]
    (out / "LABEL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
