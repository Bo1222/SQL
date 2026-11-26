"""Build CPU-only schema bundles for the 559 schema-grounding questions.

Two artifacts are produced:

``schema_bundles_retrieval.jsonl``
    A Gold-free bundle: question + Evidence + a lexically retrieved set of
    tables/columns/join edges/value hints. This is the bundle that could be
    shown to a model. It intentionally contains no Gold SQL or Gold schema path.

``schema_bundles_train.jsonl``
    A training-only, Gold-anchored bundle: it adds the Gold tables/columns and
    the Gold SQL so a later renderer can build schema-linking SFT traces. It
    must never be used as an evaluation input.

The retrieval method is deliberately simple and reproducible: IDF-weighted
table/column lexical matching, plus one-hop declared-FK neighbors. Retrieval
recall against the Gold requirements is reported in the summary, but Gold is
not used to construct the retrieval bundle.
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
    compact_schema_lines,
    dialect_columns,
    dialect_table_names,
    extract_evidence_filters,
    gold_join_edges,
    gold_requirements,
    quote_ident,
    rank_tables,
    read_jsonl,
    resolve_filter_column,
    select_columns,
    text_quoted_literals,
    write_jsonl,
)


def display_ident(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][\w]*", name) else quote_ident(name)


def columns_payload(catalog: SchemaCatalog, table: str,
                    keep: list[str] | None = None) -> list[dict]:
    info = catalog.table(table)
    if not info:
        return []
    wanted = set(keep) if keep is not None else None
    return [{
        "name": col["name"],
        "type": col["type"],
        "pk": col["pk"],
    } for col in info["columns"] if wanted is None or col["name"] in wanted]


def value_hints(question: str, evidence: str, catalog: SchemaCatalog) -> dict:
    literals = text_quoted_literals(question)
    literals += [value for value in text_quoted_literals(evidence)
                 if value not in literals]
    filters = []
    for item in extract_evidence_filters(evidence):
        table, column = resolve_filter_column(catalog, item)
        filters.append({
            "table_hint": item.get("table_hint"),
            "column": item.get("column"),
            "op": item.get("op"),
            "value": item.get("value"),
            "resolved_table": table,
            "resolved_column": column,
        })
    return {"quoted_literals": literals, "evidence_filters": filters}


def normalize_edge(edge: dict, source: str | None = None) -> dict:
    return {
        "left_table": edge.get("left_table", edge.get("from_table")),
        "left_column": edge.get("left_column", edge.get("from_column")),
        "right_table": edge.get("right_table", edge.get("to_table")),
        "right_column": edge.get("right_column", edge.get("to_column")),
        "source": source or edge.get("source", "candidate"),
    }


def edge_lines(edges: list[dict]) -> list[str]:
    lines = []
    for raw in edges:
        edge = normalize_edge(raw)
        left = f"{display_ident(edge['left_table'])}.{display_ident(edge['left_column'])}"
        right = f"{display_ident(edge['right_table'])}.{display_ident(edge['right_column'])}"
        lines.append(f"JOIN {left} = {right}  -- {edge.get('source', 'candidate')}")
    return lines


def table_payload(catalog: SchemaCatalog, tables: list[str],
                  selected_columns: dict[str, list[str] | None]) -> list[dict]:
    payload = []
    for table in tables:
        info = catalog.table(table)
        if not info:
            continue
        payload.append({
            "name": info["name"],
            "columns": columns_payload(catalog, table, selected_columns.get(table)),
        })
    return payload


def retrieval_join_edges(catalog: SchemaCatalog, selected: set[str]) -> list[dict]:
    edges = []
    seen = set()
    for edge in catalog.fk_edges:
        if edge["from_table"] in selected and edge["to_table"] in selected:
            key = tuple(sorted([
                (edge["from_table"].lower(), edge["from_column"].lower()),
                (edge["to_table"].lower(), edge["to_column"].lower()),
            ]))
            if key not in seen:
                seen.add(key)
                edges.append(normalize_edge(edge, "declared_fk"))
    for edge in catalog.name_match_edges():
        if edge["left_table"] in selected and edge["right_table"] in selected:
            key = tuple(sorted([
                (edge["left_table"].lower(), edge["left_column"].lower()),
                (edge["right_table"].lower(), edge["right_column"].lower()),
            ]))
            if key not in seen:
                seen.add(key)
                edges.append(normalize_edge(edge, "name_match"))
    return edges


def resolve_gold_edges(catalog: SchemaCatalog, gold_sql: str,
                       gold_tables: list[str]) -> list[dict]:
    edges = gold_join_edges(gold_sql, set(catalog.table_lookup))
    if edges:
        return edges
    for left, right in zip(gold_tables, gold_tables[1:]):
        path = catalog.fk_path(left, right)
        for edge in path:
            edges.append(normalize_edge(edge, "declared_fk_path"))
    return edges


def build_one(record: dict, audit_row: dict, catalog: SchemaCatalog,
              top_k: int, max_tables: int, max_columns: int) -> tuple[dict, dict, dict]:
    question = record.get("question", "") or ""
    evidence = record.get("evidence", "") or ""
    gold_sql = record.get("gold_sql", "") or ""
    db_path = record.get("db_path", "") or ""
    try:
        requirements = gold_requirements(db_path, gold_sql)
    except Exception as exc:  # noqa: BLE001
        requirements = {"ok": False, "error": str(exc), "tables": [], "columns": []}
    gold_tables = dialect_table_names(requirements, catalog)
    gold_columns = dialect_columns(requirements, catalog)
    gold_edges = resolve_gold_edges(catalog, gold_sql, gold_tables)

    selected, scores = rank_tables(
        question, evidence, catalog, top_k=top_k, max_tables=max_tables)
    selected_columns = {
        table: select_columns(catalog, table, question, evidence, max_columns,
                              peer_tables=set(selected))
        for table in selected
    }
    retrieval_edges = retrieval_join_edges(catalog, set(selected))
    retrieval_text = compact_schema_lines(catalog, selected, selected_columns)
    retrieval_text += edge_lines(retrieval_edges)
    hints = value_hints(question, evidence, catalog)

    retrieval = {
        "question_id": record.get("id"),
        "db_id": record.get("db_id"),
        "question": question,
        "evidence": evidence,
        "schema": {
            "tables": table_payload(catalog, selected, selected_columns),
            "join_edges": retrieval_edges,
            "text": "\n".join(retrieval_text),
        },
        "value_hints": hints,
        "retrieval": {
            "method": "idf_lexical_table_columns_plus_one_hop_fk",
            "catalog_tables": len(catalog.tables),
            "selected_tables": selected,
            "table_scores": {table: scores.get(table, 0.0) for table in selected},
            "top_k": top_k,
            "max_tables": max_tables,
        },
        "training_only": False,
    }

    anchored_tables = list(selected)
    for table in gold_tables:
        if table not in anchored_tables:
            anchored_tables.append(table)
    anchored_columns: dict[str, list[str] | None] = {}
    for table in anchored_tables:
        if table in gold_tables:
            anchored_columns[table] = None  # all columns for training targets
        else:
            anchored_columns[table] = selected_columns.get(table)
    anchored_text = compact_schema_lines(catalog, anchored_tables, anchored_columns)
    anchored_text += edge_lines(gold_edges)
    anchored_text += edge_lines(retrieval_edges)
    training = {
        "question_id": record.get("id"),
        "db_id": record.get("db_id"),
        "question": question,
        "evidence": evidence,
        "schema": {
            "tables": table_payload(catalog, anchored_tables, anchored_columns),
            "join_edges": gold_edges + retrieval_edges,
            "text": "\n".join(anchored_text),
        },
        "value_hints": hints,
        "gold": {
            "sql": gold_sql,
            "tables": gold_tables,
            "columns": gold_columns,
            "join_edges": gold_edges,
            "requirements_ok": requirements.get("ok"),
            "requirements_error": requirements.get("error"),
        },
        "schema_linking_trace": [
            {"step": "identify_tables", "tables": gold_tables},
            {"step": "verify_columns", "columns": gold_columns},
            {"step": "construct_joins", "join_edges": gold_edges},
            {"step": "write_sql", "sql": gold_sql},
        ],
        "training_only": True,
        "source": "gold_anchored_training_target",
    }

    selected_table_set = set(selected)
    gold_table_covered = all(table in selected_table_set for table in gold_tables)
    gold_column_covered = all(
        item["table"] in selected_table_set
        and item["column"] in (selected_columns.get(item["table"]) or [])
        for item in gold_columns
    )
    retrieval_edge_keys = {
        tuple(sorted([
            (edge["left_table"].lower(), edge["left_column"].lower()),
            (edge["right_table"].lower(), edge["right_column"].lower()),
        ])) for edge in retrieval_edges
    }
    gold_edge_covered = all(
        tuple(sorted([
            (edge["left_table"].lower(), edge["left_column"].lower()),
            (edge["right_table"].lower(), edge["right_column"].lower()),
        ])) in retrieval_edge_keys
        for edge in gold_edges
    )
    eval_row = {
        "question_id": record.get("id"),
        "db_id": record.get("db_id"),
        "category": audit_row.get("category"),
        "gold_tables": gold_tables,
        "gold_columns": gold_columns,
        "gold_join_edges": gold_edges,
        "selected_tables": selected,
        "selected_columns": selected_columns,
        "gold_table_covered": gold_table_covered,
        "gold_column_covered": gold_column_covered,
        "gold_join_covered": gold_edge_covered,
        "fully_covered": gold_table_covered and gold_column_covered,
        "requirements_ok": requirements.get("ok"),
        "retrieval_text_chars": len(retrieval["schema"]["text"]),
    }
    return retrieval, training, eval_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts",
                        default="reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl")
    parser.add_argument("--audit",
                        default="reports/m6/sft_first_20261120/sft_allzero_audit.json")
    parser.add_argument("--out", default="reports/m6/sft_first_20261120/step0_step1/schema_bundles")
    parser.add_argument("--category", default="schema_column_grounding")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-tables", type=int, default=9)
    parser.add_argument("--max-columns", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit_rows = [row for row in
                  json.loads(Path(args.audit).read_text())["questions"]
                  if row.get("category") == args.category]
    audit = {row["question_id"]: row for row in audit_rows}
    ids = sorted(audit)
    if args.limit:
        ids = ids[:args.limit]
    wanted = set(ids)
    meta: dict[str, dict] = {}
    for row in read_jsonl(args.rollouts):
        qid = row["question_id"]
        if qid in wanted and qid not in meta:
            meta[qid] = row.get("record") or {}

    catalogs: dict[str, SchemaCatalog | Exception] = {}
    retrieval_rows, training_rows, eval_rows = [], [], []
    for index, qid in enumerate(ids, 1):
        record = meta.get(qid)
        if not record or not Path(record.get("db_path", "")).exists():
            continue
        db_path = record["db_path"]
        if db_path not in catalogs:
            try:
                catalogs[db_path] = SchemaCatalog(db_path)
            except Exception as exc:  # noqa: BLE001
                catalogs[db_path] = exc
        catalog = catalogs[db_path]
        if isinstance(catalog, Exception):
            continue
        retrieval, training, eval_row = build_one(
            record, audit[qid], catalog, args.top_k, args.max_tables, args.max_columns)
        retrieval_rows.append(retrieval)
        training_rows.append(training)
        eval_rows.append(eval_row)
        if index % 50 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {qid} tables={len(retrieval['retrieval']['selected_tables'])}",
                  flush=True)

    write_jsonl(out / "schema_bundles_retrieval.jsonl", retrieval_rows)
    write_jsonl(out / "schema_bundles_train.jsonl", training_rows)
    write_jsonl(out / "retrieval_eval.jsonl", eval_rows)

    table_cov = sum(row["gold_table_covered"] for row in eval_rows)
    column_cov = sum(row["gold_column_covered"] for row in eval_rows)
    join_cov = sum(row["gold_join_covered"] for row in eval_rows)
    fully = sum(row["fully_covered"] for row in eval_rows)
    missing_tables = Counter()
    missing_columns = Counter()
    for row in eval_rows:
        selected = set(row["selected_tables"])
        for table in row["gold_tables"]:
            if table not in selected:
                missing_tables[table] += 1
        for item in row["gold_columns"]:
            if item["table"] not in selected:
                missing_tables[item["table"]] += 1
            elif item["column"] not in (row["selected_columns"].get(item["table"]) or []):
                missing_columns[f"{item['table']}.{item['column']}"] += 1
    summary = {
        "questions": len(eval_rows),
        "category": args.category,
        "top_k": args.top_k,
        "max_tables": args.max_tables,
        "max_columns": args.max_columns,
        "retrieval_method": "idf_lexical_table_columns_plus_one_hop_fk",
        "gold_table_covered": table_cov,
        "gold_table_coverage": round(table_cov / len(eval_rows), 4) if eval_rows else None,
        "gold_column_covered": column_cov,
        "gold_column_coverage": round(column_cov / len(eval_rows), 4) if eval_rows else None,
        "gold_join_covered": join_cov,
        "gold_join_coverage": round(join_cov / len(eval_rows), 4) if eval_rows else None,
        "fully_covered": fully,
        "fully_covered_rate": round(fully / len(eval_rows), 4) if eval_rows else None,
        "requirements_ok": sum(row["requirements_ok"] for row in eval_rows),
        "missing_tables_top": dict(missing_tables.most_common(20)),
        "missing_columns_top": dict(missing_columns.most_common(20)),
        "avg_selected_tables": round(
            sum(len(row["selected_tables"]) for row in eval_rows) / len(eval_rows), 2)
        if eval_rows else None,
        "avg_retrieval_text_chars": round(
            sum(row["retrieval_text_chars"] for row in eval_rows) / len(eval_rows), 1)
        if eval_rows else None,
        "caveat": "Retrieval bundle is Gold-free; train bundle is Gold-anchored and training-only.",
    }
    (out / "schema_bundle_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Step 1 CPU: schema bundles for the 559 schema-grounding questions", "",
        f"Questions: {summary['questions']}. Retrieval: top_k={args.top_k}, "
        f"max_tables={args.max_tables}, max_columns={args.max_columns}, "
        "IDF lexical tables/columns + one-hop declared FK.", "",
        "## Gold coverage of the Gold-free retrieval bundle", "",
        "| item | covered | rate |", "|---|---:|---:|",
        f"| Gold tables | {table_cov} | {summary['gold_table_coverage']:.2%} |",
        f"| Gold columns | {column_cov} | {summary['gold_column_coverage']:.2%} |",
        f"| Gold join edges | {join_cov} | {summary['gold_join_coverage']:.2%} |",
        f"| Fully covered (tables+columns) | {fully} | {summary['fully_covered_rate']:.2%} |",
        "", "Coverage is a retrieval-recall diagnostic, not an EX score. "
        "The retrieval bundle does not contain Gold SQL or Gold table lists.",
        "", "## Bundle size", "",
        f"- average selected tables: {summary['avg_selected_tables']}",
        f"- average retrieval schema text chars: {summary['avg_retrieval_text_chars']}",
        f"- requirements ok: {summary['requirements_ok']}/{summary['questions']}",
        "", "## Most frequently missed Gold tables", "",
        "| table | questions |", "|---|---:|",
    ]
    for table, count in missing_tables.most_common(15):
        lines.append(f"| {table} | {count} |")
    lines += [
        "", "## Artifacts", "",
        "- `schema_bundles_retrieval.jsonl`: Gold-free, model-facing candidate bundle.",
        "- `schema_bundles_train.jsonl`: Gold-anchored training target; never an eval input.",
        "- `retrieval_eval.jsonl`: per-question coverage diagnostics.",
        "- `schema_bundle_summary.json`: machine-readable summary.",
        "", "## Caveat", "",
        "The train bundle contains Gold SQL and Gold schema paths. It is for building",
        "training traces only. The retrieval bundle is the only file that may be shown",
        "to a model at inference/evaluation time.", "",
    ]
    (out / "SCHEMA_BUNDLE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
