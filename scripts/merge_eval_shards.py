"""Merge sharded `eval/evaluate_agent.py` outputs into one dev200 result."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--expected-n", type=int, default=200)
    args = parser.parse_args()

    records, summaries = [], []
    seen = set()
    for path in args.shard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        record_source = data.get("records")
        if isinstance(record_source, str):
            shard_records = list(read_jsonl(record_source))
        else:
            shard_records = record_source or []
        for record in shard_records:
            if record["id"] in seen:
                raise SystemExit(f"duplicate record id: {record['id']}")
            seen.add(record["id"])
            records.append(record)
        summaries.append({k: v for k, v in data.items() if k != "records"})

    if args.expected_n and len(records) != args.expected_n:
        raise SystemExit(f"expected {args.expected_n} records, got {len(records)}")

    # Refuse to merge shards that were produced with different evaluation
    # settings: a mixed merge silently invalidates the paired comparison.
    config_keys = (
        "backend", "engine", "model", "prompt_mode", "prompt_version",
        "observation_mode", "retrieval_version", "max_context",
        "max_model_len", "chat_template", "stop_after_action",
        "force_first_schema_lookup", "enable_value_lookup",
        "data", "data_sha256", "num_shards",
        "max_rounds", "max_tools", "max_schema_lookups", "max_sql_executes",
        "max_observation_tokens", "observation_budget", "tool_timeout",
        "model_config_sha256",
    )
    for key in config_keys:
        present = [i for i, item in enumerate(summaries) if key in item]
        if present and len(present) != len(summaries):
            raise SystemExit(f"config field {key} missing from some shards")
        values = {json.dumps(summaries[i].get(key), sort_keys=True) for i in present}
        if len(values) > 1:
            raise SystemExit(f"shard config mismatch for {key}: {values}")
    shard_indices = [item.get("shard_index") for item in summaries]
    if any(index is not None for index in shard_indices):
        if None in shard_indices:
            raise SystemExit(f"shard_index missing from some shards: {shard_indices}")
        if len(set(shard_indices)) != len(shard_indices):
            raise SystemExit(f"duplicate shard indices: {shard_indices}")
        num_shards = summaries[0].get("num_shards")
        if num_shards and any(not 0 <= index < num_shards for index in shard_indices):
            raise SystemExit(f"shard index outside 0..{num_shards - 1}: {shard_indices}")

    n = len(records)
    summary = {
        "backend": summaries[0].get("backend"),
        "engine": summaries[0].get("engine"),
        "model": summaries[0].get("model"),
        "prompt_mode": summaries[0].get("prompt_mode"),
        "prompt_version": summaries[0].get("prompt_version"),
        "observation_mode": summaries[0].get("observation_mode"),
        "retrieval_version": summaries[0].get("retrieval_version"),
        "max_context": summaries[0].get("max_context"),
        "max_model_len": summaries[0].get("max_model_len"),
        "max_rounds": summaries[0].get("max_rounds"),
        "max_tools": summaries[0].get("max_tools"),
        "max_schema_lookups": summaries[0].get("max_schema_lookups"),
        "max_sql_executes": summaries[0].get("max_sql_executes"),
        "max_observation_tokens": summaries[0].get("max_observation_tokens"),
        "observation_budget": summaries[0].get("observation_budget"),
        "tool_timeout": summaries[0].get("tool_timeout"),
        "model_config_sha256": summaries[0].get("model_config_sha256"),
        "chat_template": summaries[0].get("chat_template"),
        "stop_after_action": summaries[0].get("stop_after_action"),
        "force_first_schema_lookup": summaries[0].get("force_first_schema_lookup"),
        "enable_value_lookup": summaries[0].get("enable_value_lookup"),
        "n": n,
        "data": summaries[0].get("data"),
        "data_sha256": summaries[0].get("data_sha256"),
        "num_shards": summaries[0].get("num_shards"),
        "source_shard_indices": shard_indices,
        "records": args.records_jsonl,
        "ex": sum(1 for r in records if r["correct"]) / max(n, 1),
        "valid": sum(1 for r in records if r["valid"]) / max(n, 1),
        "infra": sum(1 for r in records if r["infrastructure_error"]),
        "no_answer": sum(1 for r in records if r["finish_reason"] != "answer_submitted"),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
        "avg_tool_calls": sum(r["tool_calls"] for r in records) / max(n, 1),
        "avg_schema_lookups": sum(r["schema_lookups"] for r in records) / max(n, 1),
        "avg_sql_executes": sum(r["sql_executes"] for r in records) / max(n, 1),
        "avg_response_tokens": sum(r["response_tokens"] for r in records) / max(n, 1),
        "avg_observation_tokens": sum(r["observation_tokens"] for r in records) / max(n, 1),
        "avg_latency_ms": sum(r["latency_ms"] for r in records) / max(n, 1),
        "error_types": dict(Counter(r["error_type"] for r in records)),
        "retrieval": {
            "table_recall": sum(r["usage"]["table_recall"] for r in records) / max(n, 1),
            "column_recall": sum(r["usage"]["column_recall"] for r in records) / max(n, 1),
            "join_recall": sum(r["usage"]["join_recall"] for r in records) / max(n, 1),
            "lookup_rate": sum(1 for r in records if r["usage"]["looked_up"]) / max(n, 1),
            "empty_rate": sum(1 for r in records if r["usage"]["looked_up"]
                              and not r["usage"]["tables"]) / max(n, 1),
        },
        "visible_observation": {
            "lookup_observations": sum(r["visible_lookup_observations"] for r in records),
            "relationship_observations": sum(r["visible_relationship_observations"] for r in records),
            "fallback_observations": sum(r["fallback_observations"] for r in records),
            "relationship_rate": (
                sum(r["visible_relationship_observations"] for r in records)
                / max(sum(r["visible_lookup_observations"] for r in records), 1)),
        },
        "shards": [{"path": path, "n": summaries[index].get("n")}
                   for index, path in enumerate(args.shard)],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.records_jsonl:
        records_path = Path(args.records_jsonl)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("n", "ex", "valid", "infra", "no_answer", "retrieval",
                       "visible_observation", "error_types")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
