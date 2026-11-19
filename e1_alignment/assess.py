"""E1-06 assessment: required dev200 metrics and paired transitions."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.gold_extraction import gold_columns, gold_join_keys, gold_tables
from eval.paired_statistics import bootstrap_ci, mcnemar
from reward.bird_ex_reward import score_sql

SCHEMA_ERRORS = {"no_such_column", "no_such_table", "ambiguous_column", "schema_error"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def visible_context(record):
    tables, columns, edges = set(), set(), set()
    for event in record.get("tool_events") or []:
        if event.get("action") != "schema_lookup":
            continue
        metadata = (event.get("result") or {}).get("retrieval_metadata") or {}
        for table in metadata.get("matched_tables") or []:
            tables.add(str(table).lower())
        for column in metadata.get("matched_columns") or []:
            columns.add((str(column.get("table") or "").lower(),
                         str(column.get("column") or "").lower()))
        for rel in metadata.get("relationships") or []:
            edges.add((str(rel.get("from_table") or "").lower(),
                       str(rel.get("from_column") or "").lower(),
                       str(rel.get("to_table") or "").lower(),
                       str(rel.get("to_column") or "").lower()))
    return {"tables": tables, "columns": columns, "edges": edges}


def outcome(record):
    if record.get("correct"):
        return "correct"
    error = str(record.get("error_type") or "")
    if error in SCHEMA_ERRORS:
        return "schema_error"
    if error == "syntax_error":
        return "syntax_error"
    if error == "timeout":
        return "timeout"
    if record.get("valid"):
        return "semantic_wrong"
    return "no_answer"


def summarize(records, label):
    n = len(records)
    first_ex = 0
    first_total = 0
    repair_den = repair_ok = 0
    error_counts = Counter()
    lookup_episodes = grounded_any = grounded_all = no_tables = 0
    tables_complete = columns_complete = joins_complete = 0
    outcomes = Counter()
    for record in records:
        outcomes[outcome(record)] += 1
        first_sql = record.get("first_sql")
        if first_sql:
            first_total += 1
            scored = score_sql(record["db_path"], first_sql, record["gold_sql"], timeout=30)
            first_ex += int(bool(scored.get("correct")))
        executions = record.get("executions") or []
        if executions and (executions[0].get("result") or {}).get("status") != "ok":
            repair_den += 1
            repair_ok += int(bool(record.get("correct")))
        for execution in executions:
            error = (execution.get("result") or {}).get("error_type")
            if error:
                error_counts[str(error)] += 1
        lookup_events = [event for event in record.get("tool_events") or []
                         if event.get("action") == "schema_lookup"]
        if lookup_events:
            lookup_episodes += 1
            visible = visible_context(record)
            final_sql = record.get("final_sql") or ""
            tables = {str(table).lower() for table in
                      (gold_tables(final_sql, record["db_path"]) if final_sql else set())}
            columns = {(str(table).lower(), str(column).lower())
                       for table, column in (gold_columns(final_sql, record["db_path"], tables)
                                             if final_sql else set())}
            joins = {(str(a).lower(), str(ca).lower(), str(b).lower(), str(cb).lower())
                     for a, ca, b, cb in (gold_join_keys(final_sql, record["db_path"], tables)
                                          if final_sql else set())}
            joins_visible = all(edge in visible["edges"] or
                                (edge[2], edge[3], edge[0], edge[1]) in visible["edges"]
                                for edge in joins)
            tables_ok = tables <= visible["tables"]
            columns_ok = columns <= visible["columns"]
            tables_complete += int(tables_ok)
            columns_complete += int(columns_ok)
            joins_complete += int(joins_visible)
            if not tables:
                no_tables += 1
            grounded_any += int(bool(tables & visible["tables"]) or not tables)
            grounded_all += int(tables_ok and columns_ok and joins_visible)
    return {
        "label": label, "n": n,
        "ex": sum(1 for record in records if record.get("correct")) / max(n, 1),
        "valid_rate": sum(1 for record in records if record.get("valid")) / max(n, 1),
        "first_sql_ex": first_ex / max(first_total, 1), "first_sql_cases": first_total,
        "repair_cases": repair_den, "repair_success": repair_ok,
        "repair_rate": repair_ok / max(repair_den, 1),
        "no_such_column_episodes": sum(1 for record in records
                                       if record.get("error_type") == "no_such_column"),
        "no_such_table_episodes": sum(1 for record in records
                                      if record.get("error_type") == "no_such_table"),
        "execution_error_types": dict(error_counts),
        "outcomes": dict(outcomes),
        "lookup_rate": lookup_episodes / max(n, 1),
        "schema_observation_use": {
            "denominator_lookup_episodes": lookup_episodes,
            "grounded_all_rate": grounded_all / max(lookup_episodes, 1),
            "grounded_any_rate": grounded_any / max(lookup_episodes, 1),
            "tables_complete_rate": tables_complete / max(lookup_episodes, 1),
            "columns_complete_rate": columns_complete / max(lookup_episodes, 1),
            "joins_complete_rate": joins_complete / max(lookup_episodes, 1),
            "final_sql_without_tables": no_tables,
            "rule": "grounded_all = every table/column/JOIN key parsed from final SQL appears in "
                    "the union of E1 schema_lookup observations; SQL with no parsed table counts as grounded",
        },
        "avg_tool_calls": sum(record.get("tool_calls", 0) for record in records) / max(n, 1),
        "avg_schema_lookups": sum(record.get("schema_lookups", 0) for record in records) / max(n, 1),
        "avg_response_tokens": sum(record.get("response_tokens", 0) for record in records) / max(n, 1),
        "avg_observation_tokens": sum(record.get("observation_tokens", 0) for record in records) / max(n, 1),
        "finish_reasons": dict(Counter(record.get("finish_reason") for record in records)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="reports/m6/e1_alignment/dev200_sft.jsonl")
    p.add_argument("--v2", default="reports/m6/e1_alignment/dev200_v2.jsonl")
    p.add_argument("--out", default="reports/m6/e1_alignment/assessment.json")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    sft = {record["id"]: record for record in read_jsonl(a.sft)}
    v2 = {record["id"]: record for record in read_jsonl(a.v2)}
    common = sorted(set(sft) & set(v2))
    if not common:
        raise SystemExit("no paired dev200 records")
    gain = [key for key in common if v2[key].get("correct") and not sft[key].get("correct")]
    loss = [key for key in common if sft[key].get("correct") and not v2[key].get("correct")]
    by_db = defaultdict(list)
    transitions = Counter()
    for key in common:
        by_db[sft[key]["db_id"]].append(int(bool(v2[key].get("correct"))) - int(bool(sft[key].get("correct"))))
        transitions[(outcome(sft[key]), outcome(v2[key]))] += 1
    delta = (len(gain) - len(loss)) / len(common)
    report = {
        "stage": "E1-06",
        "arms": {"sft": summarize(list(sft.values()), "sft"),
                 "v2": summarize(list(v2.values()), "v2")},
        "paired": {
            "n": len(common), "gain": len(gain), "loss": len(loss), "net": len(gain) - len(loss),
            "delta_ex": delta, "mcnemar_exact_p": mcnemar(len(gain), len(loss)),
            "db_bootstrap_ci95": bootstrap_ci(by_db, a.bootstrap, a.seed),
            "gain_ids": gain, "loss_ids": loss,
            "gain_control_outcomes": dict(Counter(outcome(sft[key]) for key in gain)),
            "loss_control_outcomes": dict(Counter(outcome(sft[key]) for key in loss)),
            "transitions": {f"{left}->{right}": count for (left, right), count in sorted(transitions.items())},
        },
        "gating_note": "Success requires V2 EX above the same-contract SFT control AND paired "
                       "McNemar p<0.1, with schema/recovery improvements; valid/lookup/loss alone do not pass.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"sft_ex": report["arms"]["sft"]["ex"], "v2_ex": report["arms"]["v2"]["ex"],
                      "paired": {k: report["paired"][k] for k in
                                 ("n", "gain", "loss", "delta_ex", "mcnemar_exact_p")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
