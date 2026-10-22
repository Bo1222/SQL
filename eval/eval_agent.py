"""Budgeted agent, fixed-workflow, static retrieval, and execution-vote evaluation."""
import argparse
import json
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from agent.common import choose_device, load_metadata, load_model, read_jsonl, select_rows, write_json
from agent.loop import AgentConfig, run_episode
from agent.protocol import Action, build_prompt, encode_observation, parse_action
from agent.reward import score_trajectory
from agent.rollout import HFGenerator
from agent.tools import DatabaseTools, run_sql
from agent.trajectory import Segment, Trajectory


def static_retrieval(row, tok, tools, config):
    text = json.dumps(tools.dispatch(Action("search_schema", row["question"])), ensure_ascii=False)
    base = len(tok.encode(build_prompt(row, tools.catalog(), 0), add_special_tokens=False))
    budget = max(0, config.max_prompt - base - 96)
    return tok.decode(tok.encode(text, add_special_tokens=False)[:budget], skip_special_tokens=False)


def fixed_episode(row, tok, generator, tools, config, context):
    """Fixed retrieval -> candidate SQL -> execution -> one final revision."""
    prompt = build_prompt(row, tools.catalog(), 1) + "\nFixed retrieval:\n" + context
    prompt += "\nFirst return <sql>candidate SQL</sql>. After feedback submit <answer>SQL</answer>.\n"
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > config.max_prompt:
        raise ValueError("fixed prompt exceeds prompt budget")
    trajectory = Trajectory(segments=[Segment("prompt", ids, prompt)], tool_calls=1, source="fixed_workflow")
    for turn in range(2):
        budget = min(config.max_action, config.max_generated - trajectory.generated_tokens,
                     config.max_context - len(trajectory.input_ids))
        if budget <= 0:
            break
        values = generator(trajectory.input_ids, budget)
        text = tok.decode(values, skip_special_tokens=True)
        trajectory.segments.append(Segment("action", list(values), text))
        try:
            action = parse_action(text)
        except (ValueError, TypeError):
            trajectory.invalid_actions += 1
            break
        trajectory.actions.append(action.name)
        if turn == 0 and action.name in {"execute_sql", "submit"}:
            trajectory.first_sql = action.arguments
            result = tools.dispatch(Action("execute_sql", action.arguments))
            trajectory.tool_calls += 1
            if result.get("infrastructure_error"):
                trajectory.stop_reason = "infrastructure_error"
                break
            values, obs = encode_observation(tok, {**result, "instruction": "Submit final SQL now"}, config.max_observation)
            if len(trajectory.input_ids) + len(values) >= config.max_context:
                trajectory.stop_reason = "context_budget"
                break
            trajectory.segments.append(Segment("observation", values, obs))
        elif turn == 1 and action.name == "submit":
            trajectory.final_sql = action.arguments
            trajectory.stop_reason = "submitted"
        else:
            break
    trajectory.stop_reason = trajectory.stop_reason or "invalid_fixed_workflow"
    return trajectory


def evaluate_row(row, tok, generator, tools, config, mode="agent", n=4):
    if mode == "agent":
        trajectory = run_episode(row, tok, generator, tools, config)
        return trajectory, 1
    context = static_retrieval(row, tok, tools, config)
    if mode == "fixed":
        return fixed_episode(row, tok, generator, tools, config, context), 1
    if mode == "single":
        trajectory = run_episode(row, tok, generator, tools, replace(config, max_tools=0), context)
        trajectory.tool_calls += 1  # static metadata retrieval is charged too
        return trajectory, 1
    candidates = min(n, config.max_tools - 1)
    if candidates < 1:
        raise ValueError("vote requires at least two tool slots (retrieval + execution)")
    candidate_config = replace(config, max_tools=0, max_generated=max(1, config.max_generated // candidates))
    episodes, keys, infrastructure = [], [], False
    for index in range(candidates):
        episode = run_episode(row, tok, generator, tools, candidate_config, context)
        signature = run_sql(row["db_path"], episode.final_sql, "signature") if episode.final_sql else {"ok": False}
        infrastructure |= bool(signature.get("infrastructure_error"))
        keys.append(signature.get("signature", f"invalid-{index}"))
        episodes.append(episode)
    counts = Counter(keys)
    choice = max(range(candidates), key=lambda i: (counts[keys[i]], -i))
    trajectory = episodes[choice]
    trajectory.source = "execution_vote"
    trajectory.tool_calls = 1 + candidates
    trajectory.diagnostics = {"selected_index": choice, "candidate_sqls": [t.final_sql for t in episodes],
                              "execution_signatures": keys, "selection": "execution_result_majority_without_gold"}
    if infrastructure:
        trajectory.stop_reason = "infrastructure_error"
    return trajectory, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="")
    p.add_argument("--device", choices=["auto", "cpu", "npu"], default="auto")
    p.add_argument("--mode", choices=["agent", "single", "vote", "fixed"], default="agent")
    p.add_argument("--disable-tool", action="append", choices=["search_schema", "inspect_column", "execute_sql"], default=[])
    p.add_argument("--error-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--seed", type=int, default=0)
    for key, field in AgentConfig.__dataclass_fields__.items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=field.default)
    a = p.parse_args()
    config = AgentConfig(**{k: getattr(a, k) for k in AgentConfig.__dataclass_fields__})
    if a.limit < 0 or a.n < 1 or a.temperature < 0:
        p.error("invalid evaluation budgets")
    if (a.mode == "single" and config.max_tools < 1) or (a.mode in {"fixed", "vote"} and config.max_tools < 2):
        p.error("single needs one tool slot; fixed/vote need at least two")
    import torch
    torch.manual_seed(a.seed)
    rows = read_jsonl(a.data)
    rows = select_rows(rows, a.limit, a.seed)
    if not rows:
        p.error("empty evaluation dataset")
    model, tok = load_model(a.model, a.adapter, choose_device(a.device))
    generator = HFGenerator(model, tok, a.temperature)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records_path = out.with_suffix(".jsonl")
    totals = Counter()
    started = time.monotonic()
    with records_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            begin_in, begin_out = generator.input_tokens, generator.output_tokens
            tools = DatabaseTools(row, load_metadata(row), a.disable_tool, a.error_only)
            try:
                trajectory, candidates = evaluate_row(row, tok, generator, tools, config, a.mode, a.n)
                final = score_trajectory(row, trajectory)
                initial = run_sql(row["db_path"], trajectory.first_sql, "score", 30, row["gold_sql"]) if trajectory.first_sql else {"ok": False, "correct": False}
            except ValueError as exc:
                trajectory, candidates = Trajectory(stop_reason="prompt_budget"), 0
                final = {"ok": False, "correct": False, "error": str(exc)}
                initial = {"ok": False, "correct": False}
            record = {"id": row["id"], "db_id": row["db_id"], "question": row["question"],
                      "gold_sql": row["gold_sql"], "mode": a.mode, "final": final, "initial": initial,
                      "input_tokens": generator.input_tokens - begin_in, "output_tokens": generator.output_tokens - begin_out,
                      "candidate_count": candidates, "trajectory": trajectory.to_dict()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
            totals.update({"n": 1, "correct": int(final.get("correct", False)), "valid": int(final["ok"]),
                           "initial_correct": int(initial.get("correct", False)), "initial_valid": int(initial["ok"]),
                           "repair_denominator": int(bool(trajectory.first_sql) and not initial["ok"]),
                           "repair_correct": int(bool(trajectory.first_sql) and not initial["ok"] and final.get("correct", False)),
                           "repair_valid": int(bool(trajectory.first_sql) and not initial["ok"] and final["ok"]),
                           "infrastructure": int(final.get("infrastructure_error", False)),
                           "tool_calls": trajectory.tool_calls, "input_tokens": record["input_tokens"],
                           "output_tokens": record["output_tokens"], "invalid_actions": trajectory.invalid_actions,
                           "multi_tool_episodes": int(trajectory.tool_calls >= 2),
                           "action_segments": sum(s.kind == "action" for s in trajectory.segments),
                           "observation_tokens": sum(len(s.ids) for s in trajectory.segments if s.kind == "observation"),
                           "failed_submission": int(trajectory.final_sql is None),
                           "clean_correct": int(final.get("ex_clean", False))})
            print(f"[eval] {index}/{len(rows)} EX={totals['correct']/index:.3f} valid={totals['valid']/index:.3f}", flush=True)
    summary = {**vars(a), **dict(totals), "configuration": vars(a), "sampling_n": a.n,
               "ex": totals["correct"] / len(rows), "valid_rate": totals["valid"] / len(rows),
               "initial_ex": totals["initial_correct"] / len(rows), "ex_clean": totals["clean_correct"] / len(rows),
               "repair_correct_rate": totals["repair_correct"] / max(totals["repair_denominator"], 1),
               "repair_exec_rate": totals["repair_valid"] / max(totals["repair_denominator"], 1),
               "avg_tool_calls": totals["tool_calls"] / len(rows), "elapsed_s": time.monotonic() - started,
               "format_error_rate": totals["invalid_actions"] / max(totals["action_segments"], 1),
               "submission_failure_rate": totals["failed_submission"] / len(rows),
               "records": str(records_path)}
    write_json(out, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
