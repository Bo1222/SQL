"""E1-04: gold-AST-assisted programmatic E1 demonstrations.

Gold SQL is only used offline: to extract required tables/columns/JOIN keys,
to construct the retrieval query and as the supervised assistant SQL.  It never
enters the user prompt or the tool observation.  A sample is kept only when the
real E1 observation covers every required table, column and JOIN key.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from agent.sql_episode import encode_observation
from e1_alignment.common import (RL_TRAIN, append_jsonl, read_jsonl, stratified_sample,
                                 write_json, write_jsonl)
from e1_alignment.dataset import (apply_loss_mask, dedupe_key, finalize_record,
                                  score_records)
from e1_alignment.runtime import Runtime, RuntimeContract

DEFAULT_RAW = "artifacts/m6/e1_alignment/e1_gold_assisted.raw.jsonl"
DEFAULT_OUT = "artifacts/m6/e1_alignment/e1_gold_assisted.jsonl"
DEFAULT_COVERAGE = "artifacts/m6/e1_alignment/e1_gold_ast_coverage.jsonl"
DEFAULT_SUMMARY = "reports/m6/e1_alignment/e1_gold_assisted.summary.json"


def requirements_for(row, runtime):
    from eval.gold_extraction import extract_requirements
    return extract_requirements(row["gold_sql"], row["db_path"])


def metadata_coverage(requirements, metadata_union):
    tables = {str(t).lower() for t in metadata_union["tables"]}
    columns = {(str(c["table"]).lower(), str(c["column"]).lower())
               for c in metadata_union["columns"]}
    edges = set()
    for rel in metadata_union["relationships"]:
        edges.add((str(rel.get("from_table") or "").lower(), str(rel.get("from_column") or "").lower(),
                   str(rel.get("to_table") or "").lower(), str(rel.get("to_column") or "").lower()))
    required_tables = {t.lower() for t in requirements["tables"]}
    required_columns = {tuple(c.lower().split(".", 1)) for c in requirements["columns"]}
    required_joins = []
    for item in requirements["join_keys"]:
        left, right = item.split("=")
        a, ca = left.split(".", 1)
        b, cb = right.split(".", 1)
        required_joins.append((a.lower(), ca.lower(), b.lower(), cb.lower()))
    table_hits = required_tables & tables
    column_hits = required_columns & columns
    join_hits = [edge for edge in required_joins
                 if edge in edges or (edge[2], edge[3], edge[0], edge[1]) in edges]
    return {
        "tables_required": sorted(required_tables), "tables_hit": sorted(table_hits),
        "columns_required": sorted(".".join(c) for c in required_columns),
        "columns_hit": sorted(".".join(c) for c in column_hits),
        "joins_required": ["=".join([f"{a}.{ca}", f"{b}.{cb}"]) for a, ca, b, cb in required_joins],
        "joins_hit": ["=".join([f"{a}.{ca}", f"{b}.{cb}"]) for a, ca, b, cb in join_hits],
        "tables_complete": required_tables.issubset(tables),
        "columns_complete": required_columns.issubset(columns),
        "joins_complete": len(join_hits) == len(required_joins),
    }


def run_lookup(runtime, db_id, arguments):
    text = f"<schema_lookup>{json.dumps(arguments, ensure_ascii=False)}</schema_lookup>"
    ids = runtime.tokenizer.encode(text, add_special_tokens=False)
    result = runtime.schema_tool(db_id)(arguments)
    return {"kind": "action", "ids": ids, "text": text, "action": "schema_lookup",
            "arguments": arguments, "invalid": False}, result


def observation_segment(runtime, result, tool_event_index):
    ids, text = encode_observation(runtime.tokenizer, result,
                                   runtime.contract.max_observation_tokens,
                                   mode=runtime.contract.observation_mode)
    return {"kind": "observation", "ids": ids, "text": text, "tool_event_index": tool_event_index}


def missing_requirements(requirements, union):
    tables = {str(t).lower() for t in union["tables"]}
    columns = {(str(c["table"]).lower(), str(c["column"]).lower()) for c in union["columns"]}
    edges = {(str(r.get("from_table") or "").lower(), str(r.get("from_column") or "").lower(),
              str(r.get("to_table") or "").lower(), str(r.get("to_column") or "").lower())
             for r in union["relationships"]}
    required_columns = {tuple(c.lower().split(".", 1)) for c in requirements["columns"]}
    required_joins = []
    for item in requirements["join_keys"]:
        left, right = item.split("=")
        a, ca = left.split(".", 1)
        b, cb = right.split(".", 1)
        required_joins.append((a.lower(), ca.lower(), b.lower(), cb.lower()))
    return {
        "tables": [t for t in requirements["tables"] if t.lower() not in tables],
        "columns": [".".join(c) for c in sorted(required_columns - columns)],
        "joins": ["=".join([f"{a}.{ca}", f"{b}.{cb}"]) for a, ca, b, cb in required_joins
                  if (a, ca, b, cb) not in edges and (b, cb, a, ca) not in edges],
    }


def missing_query(row, missing):
    terms = list(missing["tables"]) + [column.split(".", 1)[-1] for column in missing["columns"]]
    for item in missing["joins"]:
        for side in item.split("="):
            terms.append(side.split(".", 1)[-1])
    dedup = []
    for term in terms:
        for variant in (term, term.replace("_", " ")):
            if variant and variant not in dedup:
                dedup.append(variant)
    return ((row.get("question") or "").strip() + " " + " ".join(dedup)).strip()


def build_query(row, requirements, targeted=False):
    question = (row["question"] or "").strip()
    if not targeted:
        evidence = (row.get("evidence") or "").strip()
        return (question + (" " + evidence if evidence else "")).strip()
    terms = []
    for table in requirements["tables"]:
        terms.append(table)
    for column in requirements["columns"]:
        terms.append(column.split(".", 1)[-1])
    terms += [term.replace("_", " ") for term in terms]
    dedup = []
    for term in terms:
        if term and term not in dedup:
            dedup.append(term)
    return (question + " " + " ".join(dedup)).strip()


def attempt(row, runtime):
    requirements = requirements_for(row, runtime)
    if not requirements["tables"]:
        return None
    union = {"tables": set(), "columns": [], "relationships": [], "values": []}
    seen_columns = set()
    seen_edges = set()
    segments, tool_events, lookups = [], [], []

    selected = None
    first_table = requirements["tables"][0]
    for number in range(runtime.contract.max_schema_lookups):
        if number == 0:
            # Start from the first gold table: a real table lookup exposes all
            # columns of that table (subject to the frozen top-k) and its joins.
            candidate = {"table": first_table, "targeted": True}
        else:
            missing = missing_requirements(requirements, union)
            if missing["tables"]:
                candidate = {"table": missing["tables"][0], "targeted": True}
            else:
                candidate = {"query": missing_query(row, missing), "targeted": True}
        arguments = {k: v for k, v in candidate.items() if k in ("query", "table") and v}
        action, result = run_lookup(runtime, row["db_id"], arguments)
        lookups.append({"arguments": arguments, "result": result,
                        "gold_assisted_query": bool(candidate["targeted"])})
        index = len(tool_events)
        tool_events.append({"action": "schema_lookup", "arguments": arguments, "result": result})
        action["tool_event_index"] = index
        segments.append(action)
        segments.append(observation_segment(runtime, result, index))
        metadata = result.get("retrieval_metadata") or {}
        union["tables"] |= set(metadata.get("matched_tables") or [])
        for column in metadata.get("matched_columns") or []:
            key = (str(column.get("table")).lower(), str(column.get("column")).lower())
            if key not in seen_columns:
                seen_columns.add(key)
                union["columns"].append({"table": column.get("table"), "column": column.get("column")})
        for rel in metadata.get("relationships") or []:
            edge = (rel.get("from_table"), rel.get("from_column"), rel.get("to_table"), rel.get("to_column"))
            if edge not in seen_edges:
                seen_edges.add(edge)
                union["relationships"].append(rel)
        coverage = metadata_coverage(requirements, union)
        if coverage["tables_complete"] and coverage["columns_complete"] and coverage["joins_complete"]:
            selected = {"coverage": coverage, "lookups": lookups, "segments": segments,
                        "tool_events": tool_events, "requirements": requirements}
            break
    if selected is None:
        return None

    # Real execution of the gold SQL, then the final answer (also gold).
    from tools.sql_execute import SQLExecuteTool
    sql_tool = SQLExecuteTool(row["db_path"], timeout=runtime.contract.tool_timeout,
                              preview_rows=runtime.contract.preview_rows)
    executed = sql_tool(row["gold_sql"])
    if executed.get("status") != "ok" or executed.get("failure_origin") == "infrastructure":
        return None
    execute_text = f"<sql_execute>{row['gold_sql']}</sql_execute>"
    execute_ids = runtime.tokenizer.encode(execute_text, add_special_tokens=False)
    index = len(tool_events)
    tool_events.append({"action": "sql_execute", "arguments": row["gold_sql"], "result": executed})
    selected["segments"].append({"kind": "action", "ids": execute_ids, "text": execute_text,
                                 "action": "sql_execute", "arguments": row["gold_sql"],
                                 "invalid": False, "tool_event_index": index})
    selected["segments"].append(observation_segment(runtime, executed, index))
    answer_text = f"<answer>{row['gold_sql']}</answer>"
    answer_ids = runtime.tokenizer.encode(answer_text, add_special_tokens=False)
    selected["segments"].append({"kind": "action", "ids": answer_ids, "text": answer_text,
                                 "action": "answer", "arguments": row["gold_sql"], "invalid": False})
    selected["lookups"] = lookups
    return selected


def make_record(row, runtime, selected, fingerprint):
    prompt = runtime.prompt_for_row(row)
    prompt_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
    segments = [{"kind": "prompt", "ids": prompt_ids, "text": prompt}] + selected["segments"]
    record = {
        "id": row["id"], "db_id": row["db_id"], "question": row["question"],
        "evidence": row.get("evidence", ""), "gold_sql": row["gold_sql"], "db_path": row["db_path"],
        "prompt": prompt, "prompt_ids": prompt_ids, "segments": segments,
        "tool_events": selected["tool_events"], "final_sql": row["gold_sql"],
        "first_sql": row["gold_sql"], "finish_reason": "answer_submitted",
        "rounds": sum(1 for s in segments if s["kind"] == "action"),
        "tool_calls": len(selected["tool_events"]),
        "schema_lookups": sum(1 for e in selected["tool_events"] if e["action"] == "schema_lookup"),
        "sql_executes": sum(1 for e in selected["tool_events"] if e["action"] == "sql_execute"),
        "invalid_actions": 0, "infrastructure_error": False, "generation_error": None,
        "latency_ms": 0.0,
        "coverage": selected["coverage"], "requirements": selected["requirements"],
        "gold_queries": selected["lookups"],
    }
    score_records([record])
    apply_loss_mask(record)
    finalize_record(record, fingerprint, "e1_gold_assisted",
                    policy_checkpoint="programmatic_gold_ast",
                    gold_used_for_construction=True, autonomous_repair=False,
                    extra={"coverage": selected["coverage"],
                           "requirements": selected["requirements"]})
    record["success_type"] = "gold_assisted"
    record["excluded_reason"] = None if record.get("final_correct") else "final_ex_zero"
    record["dedupe_key"] = dedupe_key(record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(RL_TRAIN))
    p.add_argument("--limit", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-from", nargs="*", default=[],
                   help="JSONL files whose positive question_ids should not receive C samples")
    p.add_argument("--only-from", nargs="*", default=[],
                   help="restrict C to question_ids present in these JSONL files (failed A/B states)")
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--raw-out", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--coverage-out", default=DEFAULT_COVERAGE)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    a = p.parse_args()

    fingerprint = json.loads(Path(a.fingerprint).read_text(encoding="utf-8"))["fingerprint_sha256"]
    rows = read_jsonl(a.data)
    excluded = set()
    for path in a.exclude_from:
        if Path(path).is_file():
            for record in read_jsonl(path):
                if record.get("final_correct") or record.get("excluded_reason") is None:
                    excluded.add(record.get("id"))
    rows = [row for row in rows if row["id"] not in excluded]
    if a.only_from:
        allowed = set()
        for path in a.only_from:
            if Path(path).is_file():
                allowed |= {record.get("id") for record in read_jsonl(path)}
        rows = [row for row in rows if row["id"] in allowed]
    sample = stratified_sample(rows, a.limit, a.seed)
    runtime = Runtime(RuntimeContract(model_path=a.model, max_context=a.max_model_len))

    Path(a.raw_out).parent.mkdir(parents=True, exist_ok=True)
    for path in (a.raw_out, a.out, a.coverage_out):
        Path(path).write_text("", encoding="utf-8")
    started = time.monotonic()
    totals = {"attempts": 0, "covered": 0, "positive": 0,
              "by_db": defaultdict(int), "by_lookups": defaultdict(int)}
    for index, row in enumerate(sample, 1):
        totals["attempts"] += 1
        selected = attempt(row, runtime)
        if selected is None:
            append_jsonl(a.coverage_out, {"id": row["id"], "db_id": row["db_id"],
                                          "covered": False})
            continue
        totals["covered"] += 1
        record = make_record(row, runtime, selected, fingerprint)
        append_jsonl(a.coverage_out, {"id": row["id"], "db_id": row["db_id"], "covered": True,
                                      "coverage": selected["coverage"],
                                      "lookups": [entry["arguments"] for entry in selected["lookups"]]})
        append_jsonl(a.raw_out, record)
        if record.get("final_correct") and record.get("supervised_tokens"):
            totals["positive"] += 1
            totals["by_db"][row["db_id"]] += 1
            totals["by_lookups"][record["schema_lookups"]] += 1
            append_jsonl(a.out, record)
        if index % 10 == 0:
            print(json.dumps({"attempts": totals["attempts"], "covered": totals["covered"],
                              "positive": totals["positive"],
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        write_json(a.summary, {"stage": "E1-04", "limit": a.limit, "seed": a.seed,
                               "fingerprint_sha256": fingerprint,
                               "attempts": totals["attempts"], "covered": totals["covered"],
                               "positive": totals["positive"],
                               "by_db": dict(totals["by_db"]),
                               "by_lookups": {str(k): v for k, v in totals["by_lookups"].items()},
                               "elapsed_s": round(time.monotonic() - started, 1),
                               "raw_out": a.raw_out, "out": a.out,
                               "coverage_out": a.coverage_out})
    write_json(a.summary, {"stage": "E1-04", "limit": a.limit, "seed": a.seed,
                           "fingerprint_sha256": fingerprint,
                           "attempts": totals["attempts"], "covered": totals["covered"],
                           "positive": totals["positive"],
                           "by_db": dict(totals["by_db"]),
                           "by_lookups": {str(k): v for k, v in totals["by_lookups"].items()},
                           "elapsed_s": round(time.monotonic() - started, 1),
                           "raw_out": a.raw_out, "out": a.out,
                           "coverage_out": a.coverage_out, "complete": True})
    print(json.dumps({"done": True, "attempts": totals["attempts"],
                      "covered": totals["covered"], "positive": totals["positive"]}))


if __name__ == "__main__":
    main()
