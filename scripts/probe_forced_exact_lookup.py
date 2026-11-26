"""Oracle-forced exact-lookup probe (P0-2).

prepare  builds a deterministic Gold-selected ``{table, column}`` seed per
         question and records skips; it never puts Gold SQL in a prompt.
run      executes the E1.2-exact and E1.1-ignored arms with that seed as the
         first observation.
summarize pairs the arms by id and reports delivery, SQL divergence, EX, and
         the no_such_column subset.

The seed is diagnostic only and is never written to training data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.protocol import build_prompt, prompt_version_for  # noqa: E402
from agent.sql_episode import EpisodeConfig, episode_to_record, run_episode  # noqa: E402
from eval.backends import build_backend  # noqa: E402
from eval.sql_dependencies import sqlite_requirements  # noqa: E402
from reward.bird_ex_reward import score_sql  # noqa: E402
from retrieval.catalog_store import CatalogStore  # noqa: E402
from tools.sql_execute import SQLExecuteTool  # noqa: E402

SELECTION_RULE = "most_required_columns_then_lexical;first_required_column_in_catalog_order"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str)
                                    for row in rows) + "\n", encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_columns(catalog_dir, db_id):
    """Return {exact table name: [exact column names in catalog order]}."""
    store = CatalogStore(Path(catalog_dir) / db_id)
    catalog = {name: [] for name in store.table_names}
    for record in store.catalog:
        catalog.setdefault(record["table"], []).append(record["column"])
    return catalog


def select_seed(requirements, catalog):
    """Choose one deterministic Gold-selected table.column.

    requirements: output of ``sqlite_requirements`` (lowercase tables/columns).
    catalog:      exact-case table -> columns, in catalog order.
    Returns (seed, info); seed is None when the question is skipped.
    """
    info = {"selection_rule": SELECTION_RULE}
    if not requirements.get("ok"):
        info["skipped_reason"] = "requirements_not_ok"
        return None, info
    required = requirements.get("columns") or []
    info["n_required_columns"] = len(required)
    if not required:
        info["skipped_reason"] = "no_required_columns"
        return None, info
    by_table = defaultdict(set)
    for item in required:
        if "." not in item:
            continue
        table, column = item.split(".", 1)
        by_table[table].add(column)
    info["n_required_tables"] = len(by_table)
    catalog_by_lower = {table.lower(): (table, columns)
                        for table, columns in catalog.items()}
    eligible = [(table, columns) for table, columns in by_table.items()
                if table in catalog_by_lower]
    if not eligible:
        info["skipped_reason"] = "no_required_table_in_catalog"
        return None, info
    table, columns = sorted(eligible, key=lambda item: (-len(item[1]), item[0]))[0]
    exact_table, catalog_columns_list = catalog_by_lower[table]
    order = {column.lower(): index for index, column in enumerate(catalog_columns_list)}
    present = [column for column in columns if column in order]
    if not present:
        info["skipped_reason"] = "required_column_absent_from_catalog"
        return None, info
    column = sorted(present, key=lambda name: order[name])[0]
    exact_column = catalog_columns_list[order[column]]
    info.update({
        "oracle_table": exact_table,
        "oracle_column": exact_column,
        "n_required_columns_in_table": len(columns),
        "skipped_reason": None,
    })
    return {"query": exact_table, "table": exact_table, "column": exact_column}, info


def cmd_prepare(args):
    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"missing data: {data}")
    sha = sha256_file(data)
    if args.expected_data_sha256 and sha != args.expected_data_sha256:
        raise SystemExit(f"data sha256 mismatch: {sha} != {args.expected_data_sha256}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(data)
    if args.limit:
        rows = rows[:args.limit]
    seeds, skipped = [], Counter()
    for row in rows:
        try:
            catalog = catalog_columns(args.catalog_dir, row["db_id"])
        except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
            seed, info = None, {"skipped_reason": f"catalog_error:{type(exc).__name__}"}
        else:
            seed, info = select_seed(
                sqlite_requirements(row["db_path"], row["gold_sql"]), catalog)
        record = {"id": row["id"], "db_id": row["db_id"], "gold_sql": row["gold_sql"],
                  "seed": seed, **info}
        seeds.append(record)
        skipped[info.get("skipped_reason") or "seeded"] += 1
    write_jsonl(out / "seeds.jsonl", seeds)
    n_seeded = sum(1 for row in seeds if row["seed"])
    summary = {
        "ok": True,
        "data": str(data.resolve()),
        "data_sha256": sha,
        "expected_data_sha256": args.expected_data_sha256 or None,
        "sha256_match": (not args.expected_data_sha256) or sha == args.expected_data_sha256,
        "n_rows": len(seeds),
        "n_seeded": n_seeded,
        "n_skipped": len(seeds) - n_seeded,
        "outcomes": dict(skipped.most_common()),
        "selection_rule": SELECTION_RULE,
        "arms": {"E12_exact": "E1.2", "E11_ignored": "E1.1"},
        "seeds_path": str((out / "seeds.jsonl").resolve()),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def _episode_config():
    return EpisodeConfig(
        max_rounds=5, max_tools=4, max_schema_lookups=2, max_sql_executes=2,
        max_invalid_actions=2, max_observation_tokens=1024, max_action_tokens=512,
        max_context=8192, tool_timeout=10, preview_rows=3,
        observation_mode="compact_v2", max_response_tokens=4096)


def cmd_run(args):
    out = Path(args.out)
    summary_path = out / "prepare_summary.json"
    if not summary_path.is_file():
        raise SystemExit("run prepare first")
    prepare = json.loads(summary_path.read_text(encoding="utf-8"))
    if not args.seed_lookup_from_gold:
        raise SystemExit("--seed-lookup-from-gold is required for this diagnostic")
    seeds = [row for row in read_jsonl(out / "seeds.jsonl") if row.get("seed")]
    if args.limit:
        seeds = seeds[:args.limit]
    if not seeds:
        raise SystemExit("no prepared seeds")
    data_rows = {row["id"]: row for row in read_jsonl(prepare["data"])}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.engine == "vllm":
        from eval.evaluate_agent import VLLMEngine
        engine = VLLMEngine(args.model, max_model_len=args.max_model_len,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            stop_after_action=args.stop_after_action)
    else:
        from eval.evaluate_agent import HFEngine
        engine = HFEngine(args.model)
    config = _episode_config()
    prompt_version = prompt_version_for(args.prompt_mode)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    summaries = {}
    for arm in arms:
        if arm not in ("E12_exact", "E11_ignored"):
            raise SystemExit(f"unknown arm: {arm}")
        retrieval = "E1.2" if arm == "E12_exact" else "E1.1"
        records, started_errors = [], Counter()
        for seed_row in seeds:
            row = dict(data_rows[seed_row["id"]])
            metadata = json.loads(Path(
                row.get("metadata_path") or
                f"data/processed/bird_agent/metadata/{row['db_id']}.json"
            ).read_text(encoding="utf-8"))
            row["catalog_tables"] = [table["name"] for table in metadata["tables"]]
            prompt = build_prompt(row["question"], row.get("evidence", ""),
                                  "\n".join(row["catalog_tables"]),
                                  max_tools=config.max_tools, prompt_mode=args.prompt_mode)
            schema_tool = build_backend(
                "unified", row, tokenizer=tokenizer, catalog_dir=args.catalog_dir,
                token_budget=config.max_observation_tokens,
                retrieval_version=retrieval, resolve_implicit_fks=True)
            sql_tool = SQLExecuteTool(row["db_path"], timeout=config.tool_timeout,
                                      preview_rows=config.preview_rows,
                                      failure_policy="typed-v1")
            state = run_episode(row, tokenizer, engine, schema_tool, sql_tool, config,
                                prompt=prompt, seed_lookup=seed_row["seed"])
            record = episode_to_record(state, row)
            scored = score_sql(row["db_path"], record["final_sql"], row["gold_sql"],
                               timeout=30, score_mode="bird_set", typed=True)
            record.update({
                "arm": arm, "retrieval_version": retrieval,
                "seed": seed_row["seed"],
                "oracle_table": seed_row.get("oracle_table"),
                "oracle_column": seed_row.get("oracle_column"),
                "n_required_columns": seed_row.get("n_required_columns"),
                "correct": scored["correct"], "valid": scored["ok"],
                "error_type": scored.get("error_type"),
                "prompt_version": prompt_version,
                "prompt_mode": args.prompt_mode,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
            if record["finish_reason"] != "answer_submitted":
                started_errors[record["finish_reason"]] += 1
            records.append(record)
        path = out / f"records_{arm}.jsonl"
        write_jsonl(path, records)
        summaries[arm] = {
            "retrieval_version": retrieval, "records": len(records), "path": str(path),
            "finish_reasons": dict(started_errors),
            "answer_rate": sum(1 for row in records
                               if row["finish_reason"] == "answer_submitted") / len(records),
        }
        print(f"[forced-probe] {arm}: {len(records)} episodes -> {path}", flush=True)
    (out / "run_summary.json").write_text(json.dumps(summaries, indent=1) + "\n",
                                          encoding="utf-8")
    return summaries


def _normalized_sql(sql):
    return " ".join(str(sql or "").split()).lower()


def _seeded_event(record):
    return next((event for event in record.get("tool_events") or []
                 if event.get("seeded")), None)


def _seed_observation_text(record):
    for segment in record.get("segments") or []:
        if segment.get("kind") == "observation":
            return segment.get("text", "")
    return ""


def _matched_singleton(meta, seed):
    matched = meta.get("matched_columns") or []
    if len(matched) != 1 or not isinstance(matched[0], dict):
        return False
    return (str(matched[0].get("table", "")).lower() == seed["table"].lower()
            and str(matched[0].get("column", "")).lower() == seed["column"].lower())


def mcnemar_one_sided(b, c):
    """P(X >= b) under X~Binomial(b+c, 0.5); tests E12 wins > E11 wins."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def achieved_mde_net(discordant, alpha=0.05, power=0.8):
    """Approximate net-win MDE (normal approximation); reported, not a gate."""
    if discordant <= 0:
        return None
    return math.ceil((1.96 + 0.84) * math.sqrt(discordant))


def _baseline_no_such_column(path):
    ids = set()
    for record in read_jsonl(path):
        kind = record.get("error_type")
        if not kind:
            kind = (record.get("score") or {}).get("error_type")
        if kind == "no_such_column":
            ids.add(record["id"])
    return ids


def cmd_summarize(args):
    out = Path(args.out)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    by_arm = {}
    for arm in arms:
        path = out / f"records_{arm}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        by_arm[arm] = {row["id"]: row for row in read_jsonl(path)}
    if "E12_exact" not in by_arm or "E11_ignored" not in by_arm:
        raise SystemExit("summarize requires E12_exact and E11_ignored")
    ids = sorted(set(by_arm["E12_exact"]) & set(by_arm["E11_ignored"]))
    delivery = defaultdict(Counter)
    obs = defaultdict(dict)
    for arm, rows in by_arm.items():
        for question_id, record in rows.items():
            event = _seeded_event(record)
            result = (event or {}).get("result") or {}
            meta = result.get("retrieval_metadata") or {}
            delivery[arm]["records"] += 1
            delivery[arm]["delivered"] += int(bool(event) and result.get("status") == "ok")
            delivery[arm]["inspection"] += int(meta.get("inspection") is True
                                               and int(meta.get("selected_columns") or 0) > 0)
            delivery[arm]["ignored"] += int(
                meta.get("inspection") is not True
                and not _matched_singleton(meta, record["seed"]))
            obs[arm][question_id] = _seed_observation_text(record)
    n = len(ids)
    obs_diff = sum(1 for question_id in ids
                   if obs["E12_exact"].get(question_id) != obs["E11_ignored"].get(question_id))
    sql_diff = sum(1 for question_id in ids
                   if _normalized_sql(by_arm["E12_exact"][question_id].get("final_sql"))
                   != _normalized_sql(by_arm["E11_ignored"][question_id].get("final_sql")))
    b = sum(1 for question_id in ids
            if by_arm["E12_exact"][question_id].get("correct")
            and not by_arm["E11_ignored"][question_id].get("correct"))
    c = sum(1 for question_id in ids
            if by_arm["E11_ignored"][question_id].get("correct")
            and not by_arm["E12_exact"][question_id].get("correct"))
    p_one_sided = mcnemar_one_sided(b, c)
    ex = {arm: sum(1 for question_id in ids if by_arm[arm][question_id].get("correct"))
          for arm in arms}
    ex_rate = {arm: (ex[arm] / n if n else None) for arm in arms}
    delivery_rate = {arm: (delivery[arm]["delivered"] / delivery[arm]["records"]
                           if delivery[arm]["records"] else None) for arm in arms}
    inspection_rate = (delivery["E12_exact"]["inspection"] / delivery["E12_exact"]["records"]
                       if delivery["E12_exact"]["records"] else None)
    ignored_rate = (delivery["E11_ignored"]["ignored"] / delivery["E11_ignored"]["records"]
                    if delivery["E11_ignored"]["records"] else None)
    sql_diff_rate = sql_diff / n if n else None
    obs_diff_rate = obs_diff / n if n else None
    e11_error_rate = (sum(1 for question_id in ids
                          if by_arm["E11_ignored"][question_id].get("finish_reason")
                          != "answer_submitted") / n) if n else None
    comparable = bool(e11_error_rate is not None and e11_error_rate <= 0.05)
    delivery_ok = bool(delivery_rate["E12_exact"] is not None
                       and delivery_rate["E12_exact"] >= 0.90
                       and inspection_rate >= 0.90 and (obs_diff_rate or 0) >= 0.90
                       and (ignored_rate or 0) >= 0.90)
    sql_ok = bool(sql_diff_rate is not None and sql_diff_rate >= 0.10)
    ex_pass = (b >= 4 and c == 0) or (b - c >= 3 and b >= 4 and c <= 1)
    stop = bool((delivery_rate["E12_exact"] or 0) < 0.90
                or (obs_diff_rate or 0) < 0.90
                or (sql_diff_rate or 0) < 0.02 or (c - b) >= 3)
    subset = {"n": 0}
    if args.baseline_records:
        baseline_ids = _baseline_no_such_column(args.baseline_records)
        subset_ids = sorted(baseline_ids & set(ids))
        subset = {
            "baseline_records": args.baseline_records,
            "n": len(subset_ids),
            "ex12": sum(1 for question_id in subset_ids
                        if by_arm["E12_exact"][question_id].get("correct")),
            "ex11": sum(1 for question_id in subset_ids
                        if by_arm["E11_ignored"][question_id].get("correct")),
            "sql_diff": sum(1 for question_id in subset_ids
                            if _normalized_sql(by_arm["E12_exact"][question_id].get("final_sql"))
                            != _normalized_sql(by_arm["E11_ignored"][question_id].get("final_sql"))),
            "ex12_correct_on_subset": sum(
                1 for question_id in subset_ids
                if by_arm["E12_exact"][question_id].get("correct")),
            "ex11_correct_on_subset": sum(
                1 for question_id in subset_ids
                if by_arm["E11_ignored"][question_id].get("correct")),
        }
    prompt_versions = sorted({row.get("prompt_version")
                              for rows in by_arm.values() for row in rows.values()})
    summary = {
        "n_pairs": n,
        "prompt_versions": prompt_versions,
        "delivery": {arm: dict(delivery[arm]) for arm in arms},
        "delivery_rate": delivery_rate,
        "e12_inspection_rate": inspection_rate,
        "e11_ignored_rate": ignored_rate,
        "observation_diff": obs_diff,
        "observation_diff_rate": obs_diff_rate,
        "sql_diff": sql_diff,
        "sql_diff_rate": sql_diff_rate,
        "ex": ex,
        "ex_rate": ex_rate,
        "discordant": {"e12_only_correct": b, "e11_only_correct": c, "net": b - c},
        "mcnemar_one_sided_p": p_one_sided,
        "achieved_mde_net_approx": achieved_mde_net(b + c),
        "e11_error_rate": e11_error_rate,
        "comparable": comparable,
        "calibration": {
            "oracle_probe50": {"before": 31, "after": 36, "wins": 6, "losses": 1,
                               "mcnemar_one_sided_p": 0.125,
                               "note": "privileged Gold-schema prompt injection; upper bound"},
        },
        "no_such_column_subset": subset,
        "gate": {
            "n_pairs_200": n == 200,
            "delivery_ok": delivery_ok,
            "e11_ignored_rate": ignored_rate,
            "sql_diff_ok": sql_ok,
            "ex_pass": ex_pass,
            "comparable": comparable,
            "stop": stop,
            "pass": (n == 200 and delivery_ok and sql_ok and ex_pass
                     and comparable and not stop),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    lines = ["# Forced exact-lookup probe", "",
             f"Paired questions: {n}", "",
             "| arm | delivered | inspection | ignored | EX |", "|---|---:|---:|---:|---:|"]
    for arm in arms:
        lines.append(f"| {arm} | {delivery_rate[arm]:.3f} | "
                     f"{inspection_rate if arm == 'E12_exact' else '-'} | "
                     f"{ignored_rate if arm == 'E11_ignored' else '-'} | {ex_rate[arm]:.3f} |")
    lines += ["", f"Observation differs: {obs_diff}/{n} ({summary['observation_diff_rate']:.3f})",
              f"SQL differs: {sql_diff}/{n} ({sql_diff_rate:.3f})",
              f"EX discordant: E12-only {b}, E11-only {c}, net {b - c}; "
              f"one-sided p={p_one_sided:.4f}",
              f"no_such_column subset: n={subset['n']}", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--catalog-dir", default="artifacts/catalog")
    prepare.add_argument("--expected-data-sha256", default="")
    prepare.add_argument("--limit", type=int, default=0)
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--arms", default="E12_exact,E11_ignored")
    run.add_argument("--prompt-mode", choices=["query", "query_v3"], default="query_v3")
    run.add_argument("--seed-lookup-from-gold", action="store_true")
    run.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    run.add_argument("--catalog-dir", default="artifacts/catalog")
    run.add_argument("--max-model-len", type=int, default=8192)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    run.add_argument("--stop-after-action", action="store_true",
                     help="match evaluator default (off); stop at the first closing tag only if set")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=cmd_run)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--out", required=True)
    summarize.add_argument("--arms", default="E12_exact,E11_ignored")
    summarize.add_argument("--baseline-records", default="")
    summarize.set_defaults(func=cmd_summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
