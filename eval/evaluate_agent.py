"""Offline agent evaluation: run episodes with a chosen retrieval backend.

Used by M4 (end-to-end validation) and M5 (E0 legacy vs E1 unified). No RL.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.sql_episode import EpisodeConfig, episode_to_record, run_episode  # noqa: E402
from agent.protocol import PROMPT_VERSION, SFT_PROMPT_VERSION, build_prompt, catalog_from_metadata  # noqa: E402
from eval.backends import build_backend  # noqa: E402
from eval.gold_extraction import extract_requirements, schema_columns  # noqa: E402
from eval.harness_audit import coverage_of  # noqa: E402
from eval.sql_dependencies import sqlite_requirements  # noqa: E402
from reward.bird_ex_reward import score_sql  # noqa: E402
from tools.sql_execute import SQLExecuteTool  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_sample(rows, limit, seed=0):
    if not limit or limit >= len(rows):
        return rows
    by_db = {}
    for row in rows:
        by_db.setdefault(row["db_id"], []).append(row)
    rng = random.Random(seed)
    picked, index = [], 0
    while len(picked) < limit:
        progressed = False
        for db in sorted(by_db):
            bucket = by_db[db]
            if index < len(bucket) and len(picked) < limit:
                picked.append(bucket[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return picked


def shard_rows(rows, shard_index, num_shards):
    """Deterministically partition rows for multi-card evaluation."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    if num_shards == 1:
        return list(rows)
    return [row for index, row in enumerate(rows)
            if index % num_shards == shard_index]


class VLLMEngine:
    def __init__(self, model_path, max_model_len=8192, gpu_memory_utilization=0.55,
                 stop_after_action=False):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.stop_after_action = stop_after_action
        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=True, max_model_len=max_model_len, trust_remote_code=False)

    def __call__(self, ids, max_new_tokens):
        kwargs = {}
        if self.stop_after_action:
            kwargs = {"stop": ["</schema_lookup>", "</sql_execute>", "</answer>"],
                      "include_stop_str_in_output": True}
        params = self.SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0, **kwargs)
        out = self.llm.generate(prompts=[{"prompt_token_ids": list(ids)}], sampling_params=params,
                                use_tqdm=False)
        return list(out[0].outputs[0].token_ids)


class HFEngine:
    def __init__(self, model_path, adapter=None, device="npu:0"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(device).eval()

    def __call__(self, ids, max_new_tokens):
        torch = self.torch
        with torch.no_grad():
            out = self.model.generate(input_ids=torch.tensor([ids]), max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return out[0, len(ids):].tolist()


def recall_from_events(events, requirements):
    """Legacy pre-encoding metadata recall; NOT model-visible coverage.

    Kept for historical comparability. Use usage_visible for actual text.
    """
    first = next((e for e in events if e["action"] == "schema_lookup"), None)
    if not first:
        return {"table_recall": 0.0, "column_recall": 0.0, "join_recall": 0.0, "looked_up": False}
    meta = (first.get("result") or {}).get("retrieval_metadata") or {}
    tables = {t.lower() for t in meta.get("matched_tables", [])}
    columns = {f"{c['table']}.{c['column']}".lower() for c in meta.get("matched_columns", [])}
    if not columns:  # legacy backend returns structured tables instead of matched_columns
        for table in (first.get("result") or {}).get("tables", []):
            for column in table.get("columns", []):
                columns.add(f"{table['name']}.{column['name']}".lower())
            tables.add(table.get("name", "").lower())
    rels = {((r.get("from_table") or "").lower(), (r.get("from_column") or "").lower(),
                          (r.get("to_table") or "").lower(), (r.get("to_column") or "").lower())
            for r in meta.get("relationships", [])}
    if not rels:  # legacy backend exposes foreign_keys per table
        for table in (first.get("result") or {}).get("tables", []):
            for fk in table.get("foreign_keys", []):
                rels.add(((table.get("name") or "").lower(), (fk.get("column") or "").lower(),
                          (fk.get("target_table") or "").lower(), (fk.get("target_column") or "").lower()))
    gold_tables = set(requirements["tables"])
    gold_columns = set(requirements["columns"])
    gold_joins = set()
    for item in requirements["join_keys"]:
        left, right = item.split("=")
        a, ca = left.split(".")
        b, cb = right.split(".")
        gold_joins.add((a, b, ca, cb))

    def join_hit(edges, keys):
        hits = 0
        for a, b, ca, cb in keys:
            if ((a, ca, b, cb) in edges) or ((b, cb, a, ca) in edges):
                hits += 1
        return hits

    return {
        "looked_up": True,
        "table_recall": len(tables & gold_tables) / max(len(gold_tables), 1),
        "column_recall": len(columns & gold_columns) / max(len(gold_columns), 1),
        "join_recall": (join_hit(rels, gold_joins) / len(gold_joins)) if gold_joins else 1.0,
        "tables": sorted(tables), "columns": sorted(columns),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/bird_agent/val.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0,
                   help="0-based shard over the sampled rows (for multi-card eval)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--backend", choices=["legacy", "unified"], default="unified")
    p.add_argument("--retrieval-version", choices=["E1", "E1.1", "E1.2"], default="E1",
                   help="E1.1 must match the version used to generate the SFT data")
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--adapter", default="")
    p.add_argument("--catalog-dir", default="artifacts/catalog")
    p.add_argument("--out", required=True)
    p.add_argument("--trace-dir", default="")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-tools", type=int, default=4)
    p.add_argument("--max-schema-lookups", type=int, default=2)
    p.add_argument("--max-sql-executes", type=int, default=2)
    p.add_argument("--max-observation-tokens", type=int, default=1024)
    p.add_argument("--observation-budget", type=int, default=1024)
    p.add_argument("--observation-mode", choices=["legacy", "compact", "compact_v2"], default="compact")
    p.add_argument("--prompt-mode", choices=["query", "sft"], default="query")
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--chat-template", action="store_true",
                   help="wrap the frozen prompt with the model's native chat template")
    p.add_argument("--stop-after-action", action="store_true",
                   help="stop generation at the first complete action closing tag")
    p.add_argument("--force-first-schema-lookup", action="store_true")
    p.add_argument("--enable-value-lookup", action="store_true")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--score-mode", choices=["project_strict", "bird_set"], default="project_strict",
                   help="Historical project row-count+set rule, or official BIRD set-only; never mix silently")
    p.add_argument("--runtime-contract", choices=["", "A-e1.2-typed-v1"], default="",
                   help="Opt-in shared A train/eval configuration; preserves historical defaults otherwise")
    a = p.parse_args()
    if a.runtime_contract:
        a.retrieval_version = "E1.2"
        a.observation_mode = "compact_v2"
        a.max_rounds, a.max_tools = 5, 4
        a.max_schema_lookups, a.max_sql_executes = 2, 2
        a.max_observation_tokens = a.observation_budget = 1024
        a.max_context = a.max_model_len = 8192
        a.score_mode, a.timeout = "bird_set", 10
        a.prompt_mode, a.backend = "query", "unified"
        a.chat_template = a.stop_after_action = a.force_first_schema_lookup = a.enable_value_lookup = False

    from transformers import AutoTokenizer
    tokenizer_path = a.model if Path(a.model).is_dir() else a.model
    tokenizer = AutoTokenizer.from_pretrained(a.adapter or tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    engine = (VLLMEngine(a.model, max_model_len=a.max_model_len,
                         gpu_memory_utilization=a.gpu_memory_utilization,
                         stop_after_action=a.stop_after_action)
              if a.engine == "vllm" else HFEngine(a.model, a.adapter or None))

    rows = stratified_sample(read_jsonl(a.data), a.limit, a.seed)
    try:
        rows = shard_rows(rows, a.shard_index, a.num_shards)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not rows:
        raise SystemExit(f"shard {a.shard_index}/{a.num_shards} is empty")
    config = EpisodeConfig(max_rounds=a.max_rounds, max_tools=a.max_tools,
                           max_schema_lookups=a.max_schema_lookups,
                           max_sql_executes=a.max_sql_executes,
                           max_observation_tokens=a.max_observation_tokens,
                           tool_timeout=a.timeout, max_context=a.max_context,
                           observation_mode=a.observation_mode,
                           force_first_schema_lookup=a.force_first_schema_lookup)
    if a.runtime_contract:
        from agent.runtime_contract import config as runtime_config
        config = runtime_config()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trace = Path(a.trace_dir) if a.trace_dir else None
    if trace:
        trace.mkdir(parents=True, exist_ok=True)
        (trace / "episodes.jsonl").unlink(missing_ok=True)

    metadata_cache, runtime_stores, records = {}, {}, []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        db_id = row["db_id"]
        if db_id not in metadata_cache:
            path = Path(row.get("metadata_path") or f"data/processed/bird_agent/metadata/{db_id}.json")
            metadata_cache[db_id] = json.loads(path.read_text(encoding="utf-8"))
        row = dict(row)
        row["catalog_tables"] = [t["name"] for t in metadata_cache[db_id]["tables"]]
        prompt = build_prompt(row["question"], row.get("evidence", ""), "\n".join(row["catalog_tables"]),
                              max_tools=a.max_tools, prompt_mode=a.prompt_mode)
        if a.runtime_contract:
            from agent.runtime_contract import load_store, prompt as runtime_prompt
            if db_id not in runtime_stores:
                runtime_stores[db_id] = load_store(a.catalog_dir, db_id, row["db_path"])
            prompt = runtime_prompt(row, runtime_stores[db_id])
        if a.chat_template:
            messages = [
                {"role": "system", "content":
                 "You are a read-only SQLite database agent. Output exactly one action and nothing else. "
                 "The body of <sql_execute> and <answer> must be raw SQL starting with SELECT or WITH, "
                 "never JSON."},
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        schema_tool = build_backend(a.backend, row, tokenizer=tokenizer, catalog_dir=a.catalog_dir,
                                    token_budget=a.observation_budget,
                                    enable_value_lookup=a.enable_value_lookup,
                                    retrieval_version=a.retrieval_version,
                                    resolve_implicit_fks=bool(a.runtime_contract))
        sql_tool = SQLExecuteTool(row["db_path"], timeout=a.timeout, preview_rows=3,
                                  failure_policy="typed-v1" if a.runtime_contract else "legacy")
        state = run_episode(row, tokenizer, engine, schema_tool, sql_tool, config, prompt=prompt)
        record = episode_to_record(state, row)
        scored = score_sql(row["db_path"], record["final_sql"], row["gold_sql"], timeout=30,
                           score_mode=a.score_mode, typed=bool(a.runtime_contract))
        requirements = extract_requirements(row["gold_sql"], row["db_path"])
        record.update({
            "backend": a.backend,
            "retrieval_version": a.retrieval_version,
            "db_path": row["db_path"],
            "correct": scored["correct"],
            "valid": scored["ok"],
            "infrastructure_error": scored.get("infrastructure_error") or record["infrastructure_error"],
            "error_type": scored.get("error_type"),
            "score_mode": a.score_mode,
            "score_implementation_version": scored.get("score_implementation_version"),
            "usage": {**recall_from_events(record["tool_events"], requirements),
                      "source": "pre_encoding_metadata_not_visible_evidence"},
            "usage_visible": coverage_of(sqlite_requirements(row["db_path"], row["gold_sql"]),
                                         record["segments"]),
            "gold_requirements": requirements,
            "response_tokens": sum(len(s["ids"]) for s in record["segments"] if s["kind"] == "action"),
            "observation_tokens": sum(len(s["ids"]) for s in record["segments"] if s["kind"] == "observation"),
            "visible_lookup_observations": sum(
                1 for s in record["segments"]
                if s["kind"] == "observation" and "Relevant Database Context" in s.get("text", "")),
            "visible_relationship_observations": sum(
                1 for s in record["segments"]
                if s["kind"] == "observation" and "Relationships:" in s.get("text", "")),
            "fallback_observations": sum(
                1 for s in record["segments"]
                if s["kind"] == "observation" and "observation truncated to fit the budget" in s.get("text", "")),
        })
        records.append(record)
        if trace:
            with (trace / "episodes.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if index % 10 == 0:
            done = sum(1 for r in records if r["correct"])
            print(f"[eval] {index}/{len(rows)} ex={done} elapsed={time.monotonic() - started:.0f}s", flush=True)

    n = len(records)
    summary = {
        "runtime_contract": a.runtime_contract,
        "score_mode": a.score_mode,
        "backend": a.backend, "engine": a.engine, "model": a.model, "n": n,
        "model_config_sha256": (
            hashlib.sha256((Path(a.model) / "config.json").read_bytes()).hexdigest()
            if (Path(a.model) / "config.json").is_file() else None),
        "data": str(a.data),
        "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "shard_index": a.shard_index,
        "num_shards": a.num_shards,
        "prompt_version": SFT_PROMPT_VERSION if a.prompt_mode == "sft" else PROMPT_VERSION,
        "prompt_mode": a.prompt_mode,
        "observation_mode": a.observation_mode,
        "retrieval_version": a.retrieval_version,
        "max_context": a.max_context, "max_model_len": a.max_model_len,
        "max_rounds": a.max_rounds, "max_tools": a.max_tools,
        "max_schema_lookups": a.max_schema_lookups,
        "max_sql_executes": a.max_sql_executes,
        "max_observation_tokens": a.max_observation_tokens,
        "observation_budget": a.observation_budget,
        "tool_timeout": a.timeout,
        "chat_template": a.chat_template, "stop_after_action": a.stop_after_action,
        "force_first_schema_lookup": a.force_first_schema_lookup,
        "enable_value_lookup": a.enable_value_lookup,
        "ex": sum(1 for r in records if r["correct"]) / max(n, 1),
        "valid": sum(1 for r in records if r["valid"]) / max(n, 1),
        "infra": sum(1 for r in records if r["infrastructure_error"]),
        "no_answer": sum(1 for r in records if r["finish_reason"] != "answer_submitted"),
        "finish_reasons": {k: sum(1 for r in records if r["finish_reason"] == k) for k in
                           {r["finish_reason"] for r in records}},
        "avg_tool_calls": sum(r["tool_calls"] for r in records) / max(n, 1),
        "avg_schema_lookups": sum(r["schema_lookups"] for r in records) / max(n, 1),
        "avg_sql_executes": sum(r["sql_executes"] for r in records) / max(n, 1),
        "avg_response_tokens": sum(r["response_tokens"] for r in records) / max(n, 1),
        "avg_observation_tokens": sum(r["observation_tokens"] for r in records) / max(n, 1),
        "avg_latency_ms": sum(r["latency_ms"] for r in records) / max(n, 1),
        "error_types": {k: sum(1 for r in records if r["error_type"] == k) for k in
                        {r["error_type"] for r in records}},
        "retrieval": {
            "table_recall": sum(r["usage"]["table_recall"] for r in records) / max(n, 1),
            "column_recall": sum(r["usage"]["column_recall"] for r in records) / max(n, 1),
            "join_recall": sum(r["usage"]["join_recall"] for r in records) / max(n, 1),
            "lookup_rate": sum(1 for r in records if r["usage"]["looked_up"]) / max(n, 1),
            "empty_rate": sum(1 for r in records if r["usage"]["looked_up"] and not r["usage"]["tables"]) / max(n, 1),
        },
        "visible_observation": {
            "lookup_observations": sum(r["visible_lookup_observations"] for r in records),
            "relationship_observations": sum(r["visible_relationship_observations"] for r in records),
            "fallback_observations": sum(r["fallback_observations"] for r in records),
            "relationship_rate": (sum(r["visible_relationship_observations"] for r in records)
                                  / max(sum(r["visible_lookup_observations"] for r in records), 1)),
        },
        "elapsed_s": round(time.monotonic() - started, 1),
        "records": str(out.with_suffix(".jsonl")),
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)[:800])
    gc.collect()


if __name__ == "__main__":
    main()

