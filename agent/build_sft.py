"""Build audited local tool demonstrations or retain successful on-policy episodes."""
import argparse
import json
import random

from agent.common import (choose_device, load_metadata, load_model, read_jsonl,
                          tokenizer_fingerprint, write_json, write_jsonl)
from agent.loop import AgentConfig, run_episode
from agent.protocol import build_prompt, encode_observation
from agent.reward import score_trajectory
from agent.tools import DatabaseTools
from agent.trajectory import Segment, Trajectory


def synthetic_trajectory(row, tokenizer, tools, variant=0):
    # Gold is offline supervision. It never enters a tool request or observation.
    trajectory = Trajectory(source="synthetic", final_sql=row["gold_sql"], stop_reason="submitted")
    def add(kind, text):
        trajectory.segments.append(Segment(kind, tokenizer.encode(text, add_special_tokens=False), text))
    add("prompt", build_prompt(row, tools.catalog()))
    def observe(result):
        ids, text = encode_observation(tokenizer, result, 384)
        trajectory.segments.append(Segment("observation", ids, text))
    if variant % 4 != 0:
        add("action", "<schema>" + row["question"] + "</schema>")
        observe(tools.search_schema(row["question"]))
        trajectory.actions.append("search_schema")
        trajectory.tool_calls += 1
    if variant % 4 == 2:
        matches = tools.search_schema(row["question"])["matches"]
        if matches:
            arguments = {"table": matches[0]["table"], "column": matches[0]["name"], "keyword": ""}
            add("action", "<values>" + json.dumps(arguments) + "</values>")
            observe(tools.inspect_column(arguments))
            trajectory.actions.append("inspect_column")
            trajectory.tool_calls += 1
    if variant % 4 == 3:
        from agent.protocol import Action
        # A genuine execution error, never a silently-wrong query with an oracle hint.
        table = tools.metadata["tables"][0]["name"].replace('"', '""')
        sql = f'SELECT __agent_missing_column__ FROM "{table}"'
        add("action", "<sql>" + sql + "</sql>")
        observe(tools.dispatch(Action("execute_sql", sql)))
        trajectory.actions.append("execute_sql")
        trajectory.tool_calls += 1
        trajectory.first_sql = sql
    add("action", "<answer>" + row["gold_sql"].strip() + "</answer>")
    trajectory.actions.append("submit")
    trajectory.first_sql = trajectory.first_sql or row["gold_sql"]
    return trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="checkpoints/bird_sql_sft")
    p.add_argument("--out", required=True)
    p.add_argument("--source", choices=["synthetic", "on-policy"], default="synthetic")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "npu"], default="auto")
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.n < 1 or a.limit < 0:
        p.error("n must be positive and limit nonnegative")
    rows = read_jsonl(a.data)
    if any(r.get("split") != "sft" for r in rows):
        p.error("Only the SFT split may supply demonstrations")
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.limit] if a.limit else rows
    if a.source == "synthetic":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        generator = None
    else:
        import torch
        from agent.rollout import HFGenerator
        torch.manual_seed(a.seed)
        model, tok = load_model(a.model, a.adapter, choose_device(a.device))
        generator = HFGenerator(model, tok, 1.)
    fingerprint = tokenizer_fingerprint(tok)
    accepted, counts = [], {"attempted": 0, "incorrect": 0, "over_budget": 0, "infrastructure": 0}
    for index, row in enumerate(rows):
        for attempt in range(a.n if generator else 1):
            tools = DatabaseTools(row, load_metadata(row))
            counts["attempted"] += 1
            trajectory = (run_episode(row, tok, generator, tools, AgentConfig(max_context=a.max_context))
                          if generator else synthetic_trajectory(row, tok, tools, index))
            if len(trajectory.input_ids) > a.max_context:
                counts["over_budget"] += 1
                continue
            result = score_trajectory(row, trajectory)
            if result.get("infrastructure_error"):
                counts["infrastructure"] += 1
            elif not result["reward"]:
                counts["incorrect"] += 1
            else:
                accepted.append({"id": row["id"] + f"_{attempt}", "parent_id": row["id"], "split": "sft",
                                 "db_id": row["db_id"], "source": trajectory.source,
                                 "tokenizer_sha256": fingerprint, "trajectory": trajectory.to_dict()})
        if (index + 1) % 20 == 0:
            print(f"[demonstrations] {index + 1}/{len(rows)} accepted={len(accepted)}", flush=True)
    write_jsonl(a.out, accepted)
    write_json(a.out + ".summary.json", {**counts, "accepted": len(accepted), "tokenizer_sha256": fingerprint,
                                          "source": a.source, "seed": a.seed})


if __name__ == "__main__":
    main()
