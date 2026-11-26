"""Build synthetic schema-linking SFT trajectories from Gold-verified bundles.

This is a CPU-only data builder. For each selected question it creates an
agent-protocol trajectory using the real E1.2 schema tool and the real SQLite
tool, with Gold SQL only as the final action target:

* v1 trace: query lookup -> SQL execute -> answer
* v3 trace: query lookup -> exact table+column lookup -> SQL execute -> answer

The model-facing prompt/observations never contain the Gold SQL. The final
answer is the Gold SQL target. Traces are marked synthetic/training-only and
must not be evaluated as model outputs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.protocol import prompt_version_for  # noqa: E402
from agent.runtime_contract import (  # noqa: E402
    FINGERPRINTS,
    contract_for,
    load_store,
    prompt_for,
)
from agent.sql_episode import encode_observation  # noqa: E402
from eval.backends import build_backend  # noqa: E402
from reward.bird_ex_reward import score_sql  # noqa: E402
from scripts.sft_hard_question_common import read_jsonl, write_jsonl  # noqa: E402
from tools.sql_execute import SQLExecuteTool  # noqa: E402


class SkipTrace(Exception):
    pass


def lookup_has_tables(result: dict) -> bool:
    metadata = result.get("retrieval_metadata") or {}
    tables = metadata.get("matched_tables")
    if tables is None:
        tables = result.get("tables")
    return bool(tables)


def lookup_text(arguments: dict) -> str:
    return "<schema_lookup>" + json.dumps(arguments, ensure_ascii=False) + "</schema_lookup>"


def sql_text(sql: str) -> str:
    return "<sql_execute>" + sql + "</sql_execute>"


def answer_text(sql: str) -> str:
    return "<answer>" + sql + "</answer>"


def select_pairs(candidates: list[dict], limit: int, max_per_db: int,
                 seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in candidates:
        groups[row["db_id"]].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rows.sort(key=lambda row: row["question_id"])
        rng.shuffle(rows)
    selected = []
    while len(selected) < limit:
        progressed = False
        for db_id in sorted(groups):
            if len(selected) >= limit:
                break
            if groups[db_id] and sum(1 for row in selected if row["db_id"] == db_id) < max_per_db:
                selected.append(groups[db_id].pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_trace(row: dict, label: dict, eval_row: dict, tokenizer, variant: str,
                catalog_dir: str, cache: dict) -> dict:
    contract_id = "A-e1.2-typed-v1" if variant == "v1" else "A-e1.2-typed-v3"
    contract = contract_for(contract_id)
    db_id = row["db_id"]
    db_path = label["db_path"]
    question = row["question"]
    evidence = row.get("evidence", "") or ""
    gold = row["gold"]
    gold_sql = (gold.get("sql") or "").strip()
    gold_columns = gold.get("columns") or []
    if not gold_sql:
        raise SkipTrace("missing_gold_sql")
    if not gold_columns:
        raise SkipTrace("missing_gold_columns")

    if db_id not in cache:
        try:
            store = load_store(catalog_dir, db_id, db_path)
            schema_tool = build_backend(
                "unified", {"db_id": db_id, "db_path": db_path},
                tokenizer=tokenizer, catalog_dir=catalog_dir,
                token_budget=contract["max_observation_tokens"],
                retrieval_version=contract["retrieval_version"],
                resolve_implicit_fks=contract["resolve_implicit_fks"])
            sql_tool = SQLExecuteTool(
                db_path, timeout=contract["tool_timeout"],
                preview_rows=contract["preview_rows"], failure_policy="typed-v1")
            cache[db_id] = (store, schema_tool, sql_tool, None)
        except Exception as exc:  # noqa: BLE001
            cache[db_id] = (None, None, None, f"{type(exc).__name__}: {exc}")
    store, schema_tool, sql_tool, error = cache[db_id]
    if error:
        raise SkipTrace(f"tool_init_error:{error}")

    prompt = prompt_for({"question": question, "evidence": evidence}, store,
                        contract_id=contract_id)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_action = contract["max_action_tokens"]
    max_observation = contract["max_observation_tokens"]
    max_context = contract["max_context"]
    segments = [{"kind": "prompt", "ids": prompt_ids, "text": prompt,
                 "supervise": False}]
    tool_events = []

    def add_action(name: str, arguments, text: str, event_index):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise SkipTrace("empty_action")
        if len(ids) > max_action:
            raise SkipTrace("action_over_budget")
        segments.append({
            "kind": "action",
            "ids": ids,
            "text": text,
            "action": name,
            "arguments": arguments,
            "supervise": True,
            "tool_event_index": event_index,
        })

    def add_observation(payload: dict, event_index: int):
        try:
            ids, text = encode_observation(
                tokenizer, payload, max_observation,
                mode=contract["observation_mode"])
        except ValueError as exc:
            raise SkipTrace(f"observation_budget:{exc}") from exc
        if len(ids) > max_observation:
            raise SkipTrace("observation_over_budget")
        segments.append({
            "kind": "observation",
            "ids": ids,
            "text": text,
            "supervise": False,
            "tool_event_index": event_index,
        })

    def do_lookup(arguments: dict):
        result = schema_tool(arguments)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise SkipTrace("lookup_not_ok")
        if not lookup_has_tables(result):
            raise SkipTrace("lookup_empty")
        event_index = len(tool_events)
        add_action("schema_lookup", arguments, lookup_text(arguments), event_index)
        add_observation(result, event_index)
        tool_events.append({"action": "schema_lookup", "arguments": arguments,
                            "result": result})

    def do_sql(sql: str):
        result = sql_tool(sql)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise SkipTrace("sql_not_ok")
        event_index = len(tool_events)
        add_action("sql_execute", sql, sql_text(sql), event_index)
        add_observation(result, event_index)
        tool_events.append({"action": "sql_execute", "arguments": sql,
                            "result": result})

    do_lookup({"query": question})
    if variant == "v3":
        first = gold_columns[0]
        exact_args = {"table": first["table"], "column": first["column"]}
        do_lookup(exact_args)
    do_sql(gold_sql)
    add_action("answer", gold_sql, answer_text(gold_sql), None)

    input_ids = [token for segment in segments for token in segment["ids"]]
    loss_mask = []
    for segment in segments:
        supervised = segment["kind"] == "action" and segment.get("supervise") is True
        loss_mask.extend([1 if supervised else 0] * len(segment["ids"]))
    if len(input_ids) != len(loss_mask):
        raise SkipTrace("ids_mask_mismatch")
    if len(input_ids) > max_context:
        raise SkipTrace("over_context")
    if not any(loss_mask):
        raise SkipTrace("no_supervised_tokens")

    for segment in segments:
        if segment["kind"] in ("prompt", "observation") and gold_sql in segment.get("text", ""):
            raise SkipTrace("gold_in_prompt_or_observation")

    scored = score_sql(label["db_path"], gold_sql, gold_sql, timeout=30,
                       score_mode=contract["score_mode"], typed=True)
    if not scored.get("correct"):
        raise SkipTrace("gold_self_score_not_correct")

    record = {
        "id": f"schema_sft::{row['question_id']}::{variant}",
        "question_id": row["question_id"],
        "db_id": db_id,
        "db_path": label["db_path"],
        "split": "sft_train_hard_schema",
        "source": "synthetic_gold_schema_linking",
        "synthetic": True,
        "training_only": True,
        "training_eligible": True,
        "final_sql": gold_sql,
        "final_ex": True,
        "score": scored,
        "provenance": {
            "source": "synthetic_gold_schema_linking",
            "trainable": True,
            "generation_mode": "synthetic_schema_linking",
            "model_path": None,
            "model_fingerprint": None,
            "runtime_fingerprint": FINGERPRINTS[contract_id],
            "retrieval_version": contract["retrieval_version"],
            "prompt_mode": contract["prompt_mode"],
            "observation_mode": contract["observation_mode"],
            "split": "sft_train_hard_schema",
            "final_ex": True,
            "final_ex_source": "gold_sql_synthetic",
            "synthetic": True,
            "source_question_id": row["question_id"],
        },
        "runtime_contract": contract_id,
        "runtime_fingerprint": FINGERPRINTS[contract_id],
        "retrieval_version": contract["retrieval_version"],
        "observation_mode": contract["observation_mode"],
        "prompt_mode": contract["prompt_mode"],
        "prompt_version": prompt_version_for(contract["prompt_mode"]),
        "label_status": label["label"]["status"],
        "schema_bundle": {
            "selected_tables": eval_row.get("selected_tables", []),
            "gold_tables": gold.get("tables", []),
            "fully_covered": eval_row.get("fully_covered"),
            "gold_column_coverage": eval_row.get("gold_column_covered"),
        },
        "supervision_policy": f"synthetic-schema-linking-{variant}-v1",
        "has_lookup": True,
        "token_count": len(input_ids),
        "supervised_tokens": sum(loss_mask),
        "masked_tokens": len(input_ids) - sum(loss_mask),
        "trajectory": {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "segments": segments,
            "tool_events": tool_events,
            "final_sql": gold_sql,
            "final_ex": True,
            "stop_reason": "answer_submitted",
            "prompt": prompt,
            "prompt_ids": prompt_ids,
        },
        "trajectory_id": f"schema_sft::{row['question_id']}::{variant}",
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles",
                        default="reports/m6/sft_first_20261120/step0_step1/schema_bundles/schema_bundles_train.jsonl")
    parser.add_argument("--labels",
                        default="reports/m6/sft_first_20261120/step0_step1/label_audit.jsonl")
    parser.add_argument("--eval",
                        default="reports/m6/sft_first_20261120/step0_step1/schema_bundles/retrieval_eval.jsonl")
    parser.add_argument("--out",
                        default="reports/m6/sft_first_20261120/step0_step1/schema_sft")
    parser.add_argument("--model",
                        default="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model")
    parser.add_argument("--catalog-dir", default="artifacts/catalog")
    parser.add_argument("--limit", type=int, default=50,
                        help="questions selected; each produces v1 and v3 traces")
    parser.add_argument("--max-per-db", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--variants", default="v1,v3")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundles = {row["question_id"]: row for row in read_jsonl(args.bundles)}
    labels = {row["question_id"]: row for row in read_jsonl(args.labels)}
    eval_rows = {row["question_id"]: row for row in read_jsonl(args.eval)}
    candidates = []
    for qid, row in bundles.items():
        label = labels.get(qid)
        eval_row = eval_rows.get(qid)
        if not label or label.get("label", {}).get("status") != "label_clean":
            continue
        if not eval_row or not eval_row.get("fully_covered") or not eval_row.get("requirements_ok"):
            continue
        if not Path(label.get("db_path", "")).exists():
            continue
        candidates.append(row)
    selected = select_pairs(candidates, args.limit, args.max_per_db, args.seed)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if any(item not in {"v1", "v3"} for item in variants):
        raise SystemExit(f"unknown variants: {variants}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache = {}
    records = []
    skipped = Counter()
    skip_examples = defaultdict(list)
    for row in selected:
        qid = row["question_id"]
        label = labels[qid]
        eval_row = eval_rows[qid]
        for variant in variants:
            try:
                record = build_trace(row, label, eval_row, tokenizer, variant,
                                     args.catalog_dir, cache)
            except SkipTrace as exc:
                skipped[f"{variant}:{exc}"] += 1
                if len(skip_examples[f"{variant}:{exc}"]) < 3:
                    skip_examples[f"{variant}:{exc}"].append(qid)
                continue
            except Exception as exc:  # noqa: BLE001
                skipped[f"{variant}:exception:{type(exc).__name__}"] += 1
                if len(skip_examples[f"{variant}:exception:{type(exc).__name__}"]) < 3:
                    skip_examples[f"{variant}:exception:{type(exc).__name__}"].append(qid)
                continue
            records.append(record)

    write_jsonl(out / "schema_linking_sft.jsonl", records)
    by_variant = Counter(row["runtime_contract"] for row in records)
    summary = {
        "questions_selected": len(selected),
        "candidates": len(candidates),
        "variants": variants,
        "traces_written": len(records),
        "traces_by_variant": dict(by_variant),
        "skipped": dict(skipped.most_common()),
        "skip_examples": {key: value for key, value in skip_examples.items()},
        "avg_token_count": round(sum(row["token_count"] for row in records) / len(records), 1)
        if records else None,
        "avg_supervised_tokens": round(
            sum(row["supervised_tokens"] for row in records) / len(records), 1)
        if records else None,
        "max_token_count": max((row["token_count"] for row in records), default=None),
        "min_token_count": min((row["token_count"] for row in records), default=None),
        "label_status": "label_clean only",
        "synthetic": True,
        "training_only": True,
        "caveat": "Synthetic Gold-answer traces; not model outputs and not evaluation data.",
    }
    (out / "schema_linking_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Schema-linking SFT traces", "",
        f"Selected questions: {len(selected)}; candidates: {len(candidates)}; "
        f"traces written: {len(records)}.", "",
        "| contract | traces |", "|---|---:|",
    ]
    for contract, count in sorted(by_variant.items()):
        lines.append(f"| {contract} | {count} |")
    lines += [
        "", "## Token statistics", "",
        f"- average tokens: {summary['avg_token_count']}",
        f"- average supervised tokens: {summary['avg_supervised_tokens']}",
        f"- min/max tokens: {summary['min_token_count']} / {summary['max_token_count']}",
        "", "## Skip reasons", "",
        "| reason | count |", "|---|---:|",
    ]
    for reason, count in skipped.most_common(30):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "", "## Trace shapes", "",
        "- v1: `schema_lookup(query)` -> `sql_execute(Gold)` -> `answer(Gold)`.",
        "- v3: `schema_lookup(query)` -> `schema_lookup(table,column)` -> "
        "`sql_execute(Gold)` -> `answer(Gold)`.",
        "", "The Gold SQL appears only in the final answer target; prompt and",
        "observations are checked for exact Gold SQL leakage. These are synthetic",
        "training-only traces; they are not evaluation predictions.", "",
    ]
    (out / "SCHEMA_LINKING_SFT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
