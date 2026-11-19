"""Paired, frozen-model diagnostic: current harness vs privileged schema context.

Gold-selected physical columns plus their real catalog descriptions are exposed
ONLY in the oracle arm. SQL, results and gold literals are never copied into
the prompt. The oracle arm is NOT a deployable harness or a test-set EX claim.
Outputs remain under reports, are explicitly non-trainable, and never enter SFT.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import build_prompt
from agent.sql_episode import EpisodeConfig, episode_to_record, run_episode
from e1_alignment.common import read_jsonl, sha256_file, stratified_sample, write_json
from eval.backends import build_backend
from eval.evaluate_agent import VLLMEngine
from eval.harness_audit import sqlite_requirements
from retrieval.catalog_store import CatalogStore
from reward.bird_ex_reward import score_sql
from tools.sql_execute import SQLExecuteTool


def oracle_context(requirements, store):
    """Catalog evidence only; use gold solely to choose table/column names."""
    selected = set(requirements["columns"])
    lines = ["Diagnostic database context (oracle-selected schema):"]
    for table in requirements["tables"]:
        lines.append("Table " + json.dumps(table, ensure_ascii=False))
        for record in store.catalog:
            key = f"{record['table']}.{record['column']}".lower()
            if key not in selected:
                continue
            item = {k: record[k] for k in ("column", "data_type", "primary_key",
                    "description", "value_description") if record.get(k) not in (None, "", False)}
            lines.append("  " + json.dumps(item, ensure_ascii=False))
    lines.append("Declared database relationships:")
    for rel in store.relationships:
        if rel["from_table"].lower() in requirements["tables"] and rel["to_table"].lower() in requirements["tables"]:
            lines.append(json.dumps(rel, ensure_ascii=False))
    return "\n".join(lines).replace("<", "\\u003c").replace(">", "\\u003e")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model")
    p.add_argument("--data", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--out", default="reports/m6/harness_audit_20261117/oracle_probe50")
    a = p.parse_args()
    started = time.monotonic()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = stratified_sample(read_jsonl(a.data), a.limit, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = EpisodeConfig(max_rounds=5, max_context=8192)
    manifest = {"diagnostic_only": True, "training_eligible": False,
                "model": a.model, "data": a.data, "data_sha256": sha256_file(a.data),
                "model_config_sha256": sha256_file(Path(a.model) / "config.json"),
                "seed": a.seed, "ids": [r["id"] for r in rows], "config": asdict(config),
                "oracle_privilege": "gold-selected schema columns, real catalog descriptions; no gold SQL/results",
                "max_seconds": a.max_seconds, "script_sha256": sha256_file(__file__)}
    write_json(out / "manifest.json", manifest)
    cache = {(r["id"], r["arm"]): r for r in read_jsonl(out / "episodes.jsonl")}
    if cache:
        raise SystemExit("refusing to mix a fresh model run with prior oracle probe outputs; use a new --out")
    engine = VLLMEngine(a.model, max_model_len=config.max_context)
    records, stores, backends = [], {}, {}
    stop_reason = None
    with (out / "episodes.jsonl").open("w") as stream:
        for index, row in enumerate(rows, 1):
            if time.monotonic() - started >= a.max_seconds:
                stop_reason = "wall_clock_budget"
                break
            db = row["db_id"]
            if db not in stores:
                stores[db] = CatalogStore(Path("artifacts/catalog") / db)
                backends[db] = build_backend("unified", row, tokenizer=tokenizer,
                                            token_budget=1024, retrieval_version="E1.1")
            row = {**row, "catalog_tables": stores[db].table_names}
            req = sqlite_requirements(row["db_path"], row["gold_sql"])
            if not req["ok"]:
                raise ValueError(f"unresolved gold requirements for {row['id']}: {req}")
            base = build_prompt(row["question"], row.get("evidence", ""),
                                "\n".join(row["catalog_tables"]), config.max_tools)
            oracle = base.rsplit("### Trajectory\n", 1)[0] + oracle_context(req, stores[db]) + "\n\n### Trajectory\n"
            if len(tokenizer.encode(oracle, add_special_tokens=False)) + config.max_action_tokens > config.max_context:
                raise ValueError("oracle prompt cannot fit; do not silently truncate diagnostic evidence")
            for arm, prompt in (("current", base), ("oracle_schema", oracle)):
                state = run_episode(row, tokenizer, engine, backends[db],
                    SQLExecuteTool(row["db_path"], timeout=config.tool_timeout), config, prompt=prompt)
                result = episode_to_record(state, row)
                scored = score_sql(row["db_path"], state["final_sql"], row["gold_sql"], timeout=30)
                result.update({"arm": arm, "diagnostic_only": True, "training_eligible": False,
                               "correct": scored["correct"], "valid": scored["ok"],
                               "score": scored, "db_path": row["db_path"]})
                records.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
            pair = records[-2:]
            print(json.dumps({"pair": index, "n": len(rows), "id": row["id"],
                              "current": pair[0]["correct"], "oracle": pair[1]["correct"],
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    paired = [(records[i], records[i + 1]) for i in range(0, len(records), 2)]
    summary = {"diagnostic_only": True, "training_eligible": False,
        "planned_pairs": len(rows), "completed_pairs": len(paired),
        "completed": len(paired) == len(rows), "stop_reason": stop_reason,
        "current_correct": sum(x["correct"] for x, y in paired),
        "oracle_correct": sum(y["correct"] for x, y in paired),
        "gain": sum(not x["correct"] and y["correct"] for x, y in paired),
        "loss": sum(x["correct"] and not y["correct"] for x, y in paired),
        "both_wrong": sum(not x["correct"] and not y["correct"] for x, y in paired),
        "oracle_valid_but_wrong": sum(y["valid"] and not y["correct"] for x, y in paired),
        "infrastructure_errors": sum(bool(x.get("infrastructure_error") or x["score"].get("infrastructure_error")) for x in records),
        "wall_seconds": round(time.monotonic() - started, 3),
        "limits": ["Small dev diagnostic; not full dev200, not an independent test.",
                   "Gold-selected schema is privileged; uplift is not a deployable harness gain.",
                   "Oracle failure is model+prompt+remaining tools, not proof all harness improvements fail."]}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
