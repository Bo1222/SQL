"""Offline harness audit; never generates training data or loads model weights.

Usage: python scripts/audit_text2sql_harness.py --out reports/m6/harness_audit
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.sql_episode import EpisodeConfig
from e1_alignment.common import read_jsonl, sha256_file, write_json
from eval.harness_audit import (AuditTokenizer, best_lookup_plan, coverage_of,
                                gold_injection, lookup_candidates,
                                materialize_observation, sqlite_requirements)
from retrieval.catalog_store import CatalogStore
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import run_sql


def source_spec(value):
    name, path = value.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", type=source_spec,
                        help="NAME=JSONL, may repeat; defaults dev200 and test1534")
    parser.add_argument("--trace", action="append", type=source_spec,
                        help="NAME=JSONL, existing trajectories; may repeat")
    parser.add_argument("--tokenizer", default="artifacts/checkpoints/step_600_merged")
    parser.add_argument("--out", default="reports/m6/harness_audit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument("--retrieval-version", default="E1.1")
    parser.add_argument("--observation-mode", default="compact")
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = args.data or [("dev200", Path("artifacts/datasets/dev200.jsonl")),
                           ("test1534", Path("data/processed/bird_agent/test.jsonl"))]
    traces = args.trace or [(name, Path("reports/m6/e1_student_only") / file) for name, file in [
        ("dev200_base", "dev200_base_e1_1.jsonl"), ("dev200_sft", "sft_lowlr_788.jsonl"),
        ("dev200_r1_step20", "rl_r1_step20.jsonl"),
        ("dev200_r3_step10", "rl_r3_step10.jsonl"),
        ("test1534_base", "test1534_base_e1_1.jsonl"),
        ("test1534_sft", "test1534_sft_lowlr_788.jsonl")]]
    tokenizer = AuditTokenizer(args.tokenizer)
    config = EpisodeConfig(max_rounds=5, max_context=8192, observation_mode=args.observation_mode)
    stores, tools, requirements, rows_by_id = {}, {}, {}, {}
    cases = []
    for name, path in sources:
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[:args.limit]
        for index, row in enumerate(rows, 1):
            db = row["db_id"]
            if db not in stores:
                stores[db] = CatalogStore(Path("artifacts/catalog") / db)
                tools[db] = SchemaLookupTool(stores[db], token_budget=1024,
                    count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                    retrieval_version=args.retrieval_version)
            row = {**row, "catalog_tables": stores[db].table_names}
            req = sqlite_requirements(row["db_path"], row["gold_sql"])
            requirements[row["id"]] = req
            rows_by_id[row["id"]] = row
            candidates = lookup_candidates(row, req) if req["ok"] else []
            candidates = [c for c in candidates if len(tokenizer.encode(
                "<schema_lookup>" + json.dumps(c, ensure_ascii=False) + "</schema_lookup>",
                add_special_tokens=False)) <= config.max_action_tokens]
            observations, retrievals = [], []
            for candidate in candidates:
                result = tools[db](candidate)
                obs = materialize_observation(tokenizer, result, config)
                observations.append(obs["text"])
                raw_req = set(req["columns"])
                meta = result.get("retrieval_metadata", {})
                selected = {f"{c['table']}.{c['column']}".lower() for c in meta.get("matched_columns", [])}
                cov = coverage_of(req, [obs])
                retrievals.append({"arguments": candidate, "tokens": obs["tokens"],
                    "metadata_required_columns": sorted(selected & raw_req),
                    "visible_required_columns": sorted(set(cov["visible_columns"]) & raw_req),
                    "wrapper_truncated": '"truncated": true' in obs["text"],
                    "metadata_overstates_columns": bool((selected & raw_req) - set(cov["visible_columns"]))})
            plan = best_lookup_plan(req, candidates, observations)
            case = {"id": row["id"], "db_id": db, "dataset": name,
                    "requirements": {k: v for k, v in req.items() if k != "schema"},
                    "gold_injection": gold_injection(row, tokenizer, config),
                    "gold_lookup_submission": gold_injection(row, tokenizer, config,
                        plan["selected_arguments"], schema_tool=tools[db]),
                    "finite_lookup_search": plan,
                    "selected_observations": [observations[i] for i in plan["selected_indices"]],
                    "retrieval_diagnostics": retrievals}
            cases.append(case)
            if index % 50 == 0:
                print(json.dumps({"dataset": name, "lookup_audited": index,
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    case_path = out / "cases.jsonl"
    with case_path.open("w") as stream:
        for c in cases:
            stream.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Gold self-score: the same sandboxed worker, database and timeout as eval.
    # Preserve raw gold errors instead of silently merging them into model EX=0.
    if not args.skip_sql:
        score_path = out / "gold_scores.jsonl"
        cached = {(r["id"], r["input_sha256"]): r for r in read_jsonl(score_path)}
        with ThreadPoolExecutor(max_workers=args.workers) as pool, score_path.open("a") as stream:
            jobs = {}
            for case in cases:
                row = rows_by_id[case["id"]]
                import hashlib
                digest = hashlib.sha256(json.dumps([row["db_path"], row["gold_sql"], 30]).encode()).hexdigest()
                case["score_input_sha256"] = digest
                if (row["id"], digest) in cached:
                    case["gold_self_score"] = cached[(row["id"], digest)]["score"]
                    continue
                job = pool.submit(run_sql, row["db_path"], row["gold_sql"], operation="score",
                                  timeout=30, gold_sql=row["gold_sql"])
                jobs[job] = case
            for done, future in enumerate(as_completed(jobs), 1):
                case = jobs[future]
                case["gold_self_score"] = future.result()
                stream.write(json.dumps({"id": case["id"], "input_sha256": case["score_input_sha256"],
                                         "score": case["gold_self_score"]}, ensure_ascii=False) + "\n")
                stream.flush()
                if done % 100 == 0:
                    print(json.dumps({"gold_scored": done, "jobs": len(jobs),
                                      "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    trace_stats = {}
    with (out / "trace_cases.jsonl").open("w") as stream:
        for name, path in traces:
            stats, errors, finish = Counter(), Counter(), Counter()
            for trace in read_jsonl(path):
                if trace["id"] not in requirements:
                    continue
                req = requirements[trace["id"]]
                cov = coverage_of(req, trace.get("segments", []))
                stats["n"] += 1
                correct, valid = bool(trace.get("correct")), bool(trace.get("valid"))
                stats["correct"] += correct
                stats["valid_but_wrong"] += valid and not correct
                stats["invalid_sql"] += not valid
                stats["infrastructure_errors"] += bool(trace.get("infrastructure_error"))
                stats["requirements_resolved"] += req["ok"]
                stats["schema_visible"] += cov["all_required_schema_visible"]
                stats["wrong_with_schema_visible"] += not correct and cov["all_required_schema_visible"]
                stats["correct_without_full_schema_visible"] += correct and not cov["all_required_schema_visible"]
                stats["wrong_missing_schema"] += not correct and req["ok"] and not cov["all_required_schema_visible"]
                stats["legacy_empty_columns_but_sqlite_reads_columns"] += bool(req["columns"]) and not trace.get("gold_requirements", {}).get("columns")
                stats["legacy_requirements_omit_sqlite_columns"] += bool(set(req["columns"]) -
                    set(trace.get("gold_requirements", {}).get("columns", [])))
                errors[str(trace.get("error_type"))] += 1
                finish[str(trace.get("finish_reason"))] += 1
                stream.write(json.dumps({"trace": name, "id": trace["id"], "db_id": trace["db_id"],
                    "correct": correct, "valid": valid, "error_type": trace.get("error_type"),
                    "coverage": cov}, ensure_ascii=False) + "\n")
            trace_stats[name] = {**stats, "errors": dict(errors), "finish": dict(finish),
                                 "file": str(path), "sha256": sha256_file(path) if path.exists() else None}

    groups = defaultdict(list)
    for c in cases:
        groups[c["dataset"]].append(c)
    summaries = {}
    for name, group in groups.items():
        scored = [c for c in group if "gold_self_score" in c]
        summaries[name] = {"n": len(group),
            "requirements_resolved": sum(c["requirements"]["ok"] for c in group),
            "gold_submitted_exactly": sum(c["gold_injection"]["submitted_exact_gold"] for c in group),
            "gold_over_action_budget": sum(c["gold_injection"]["action_tokens"] > config.max_action_tokens for c in group),
            "gold_self_scored": len(scored),
            "gold_self_correct": sum(bool(c["gold_self_score"].get("correct")) for c in scored),
            "gold_submittable_and_scores_correct": sum(c["gold_injection"]["submitted_exact_gold"] and
                bool(c["gold_self_score"].get("correct")) for c in scored),
            "lookup_plan_then_gold_submitted": sum(c["gold_lookup_submission"]["submitted_exact_gold"] for c in group),
            "finite_lookup_search_schema_complete": sum(c["finite_lookup_search"]["all_required_schema_visible"] for c in group),
            "finite_lookup_search_schema_incomplete": sum(c["requirements"]["ok"] and not c["finite_lookup_search"]["all_required_schema_visible"] for c in group),
            "cases_with_metadata_overstatement": sum(any(r["metadata_overstates_columns"] for r in c["retrieval_diagnostics"]) for c in group),
            "cases_with_wrapper_truncation": sum(any(r["wrapper_truncated"] for r in c["retrieval_diagnostics"]) for c in group),
            "gold_failures": [{"id": c["id"], "score": c["gold_self_score"]} for c in scored if not c["gold_self_score"].get("correct")]}
    code_paths = ["agent/sql_episode.py", "agent/protocol.py", "tools/schema_lookup.py", "tools/sql_worker.py",
                  "retrieval/retriever.py", "retrieval/e1_1.py", "retrieval/observation_formatter.py",
                  "retrieval/e1_2.py", "eval/sql_dependencies.py", "eval/harness_audit.py",
                  "scripts/audit_text2sql_harness.py"]
    report = {"diagnostic_only": True, "model_inference": False, "training": False,
        "limits": ["Gold injection tests submission/execution, not information sufficiency or model ability.",
                   "Finite gold-guided lookup search is not a global oracle or an EX ceiling.",
                   "Missing explicit schema does not imply impossible: SQL probes/model knowledge may supply it.",
                   "Complete schema does not guarantee semantic correctness or required value discovery.",
                   "Existing trace associations are observational, not causal harness improvement estimates."],
        "config": asdict(config), "retrieval_version": args.retrieval_version,
        "tokenizer": args.tokenizer, "tokenizer_sha256": sha256_file(Path(args.tokenizer) / "tokenizer.json"),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sources},
        "code_sha256": {p: sha256_file(p) for p in code_paths},
        "datasets": summaries, "existing_traces": trace_stats,
        "wall_seconds": round(time.monotonic() - started, 3)}
    with case_path.open("w") as stream:
        for c in cases:
            stream.write(json.dumps(c, ensure_ascii=False) + "\n")
    write_json(out / "summary.json", report)
    print(json.dumps({"datasets": summaries, "existing_traces": trace_stats,
                      "wall_seconds": report["wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
