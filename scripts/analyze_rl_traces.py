"""Analyze RL rollout traces into group and behavior metrics.

The veRL trainer logs scalar reward but not the mixed-group breakdown.  This
script joins trace ``question_id`` to the RL parquet, re-scores the final SQL
with the same BIRD EX scorer, and reports:

* n4 group mixed / all-zero / all-correct rates;
* first-action and schema-error transition behavior;
* valid-but-wrong, repair, tool calls and response lengths.

It is a task-level audit, not a replacement for dev200/test1534.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402
from reward.process_signals import extract_process_events, iter_jsonl  # noqa: E402


def load_tasks(path: str) -> dict[str, dict[str, Any]]:
    frame = pd.read_parquet(path)
    tasks: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        info = dict(row["extra_info"])
        qid = str(info.get("question_id"))
        reward = dict(row["reward_model"])
        tasks[qid] = {
            "db_path": info.get("db_path"),
            "gold_sql": reward.get("ground_truth"),
            "db_id": info.get("db_id"),
        }
    return tasks


def score_trace(trace: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    sql = trace.get("final_sql")
    db_path = task.get("db_path")
    gold_sql = task.get("gold_sql")
    if not sql or not db_path or not gold_sql:
        return {
            "correct": False,
            "valid": False,
            "valid_but_wrong": False,
            "error_type": "missing",
            "infrastructure_error": True,
        }
    scored = score_sql(db_path, sql, gold_sql)
    correct = bool(scored.get("correct"))
    valid = bool(scored.get("ok"))
    return {
        "correct": correct,
        "valid": valid,
        "valid_but_wrong": bool(valid and not correct),
        "error_type": scored.get("error_type"),
        "infrastructure_error": bool(scored.get("infrastructure_error")),
    }


def first_action(trace: dict[str, Any]) -> str:
    events = trace.get("tool_events") or []
    if events:
        return str(events[0].get("action") or "unknown")
    if trace.get("final_sql"):
        return "answer"
    return "none"


def analyze(trace_dir: str, task_file: str, limit: int = 0) -> dict[str, Any]:
    tasks = load_tasks(task_file)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    checked = 0
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        for trace in iter_jsonl(str(path)):
            if limit and checked >= limit:
                break
            qid = str(trace.get("question_id"))
            task = tasks.get(qid)
            if task is None:
                missing += 1
                continue
            checked += 1
            scored = score_trace(trace, task)
            events = extract_process_events(
                tool_events=trace.get("tool_events"),
                finish_reason=trace.get("finish_reason"),
                final_sql=trace.get("final_sql"),
                first_sql=trace.get("first_sql"),
                invalid_actions=trace.get("invalid_actions", 0),
                turns=trace.get("turns", 0),
                schema_lookups=trace.get("schema_lookups"),
                sql_executes=trace.get("sql_executes"),
                tool_calls=trace.get("tool_calls"),
            )
            groups[qid].append({
                "correct": scored["correct"],
                "valid": scored["valid"],
                "valid_but_wrong": scored["valid_but_wrong"],
                "infra": scored["infrastructure_error"],
                "first_action": first_action(trace),
                "events": events,
                "turns": int(trace.get("turns") or 0),
                "tool_calls": events.tool_calls,
                "response_tokens": int(trace.get("response_tokens") or 0),
            })
    if limit:
        groups = {qid: rows[:limit] for qid, rows in groups.items()}

    complete = {qid: rows for qid, rows in groups.items() if len(rows) == 4 and not any(r["infra"] for r in rows)}
    mixed = all_zero = all_correct = 0
    for rows in complete.values():
        n_correct = sum(1 for row in rows if row["correct"])
        if 0 < n_correct < len(rows):
            mixed += 1
        elif n_correct == 0:
            all_zero += 1
        else:
            all_correct += 1

    first_actions = Counter(row["first_action"] for rows in complete.values() for row in rows)
    transitions = Counter()
    behavior = Counter()
    for rows in complete.values():
        for row in rows:
            events = row["events"]
            behavior["valid_but_wrong"] += int(row["valid_but_wrong"])
            behavior["invalid_actions"] += events.invalid_actions
            behavior["repeat_sql"] += events.repeat_sql
            if events.schema_error_then_lookup:
                transitions["schema_error_then_lookup"] += 1
            if events.schema_error_then_answer:
                transitions["schema_error_then_answer"] += 1
            if events.schema_error_repaired:
                transitions["schema_error_repaired"] += 1
    total = sum(len(rows) for rows in complete.values())
    return {
        "trace_dir": str(trace_dir),
        "task_file": str(task_file),
        "traces_checked": checked,
        "missing_task": missing,
        "groups_total": len(groups),
        "groups_complete": len(complete),
        "groups_mixed": mixed,
        "groups_all_zero": all_zero,
        "groups_all_correct": all_correct,
        "mixed_rate": mixed / len(complete) if complete else None,
        "all_zero_rate": all_zero / len(complete) if complete else None,
        "all_correct_rate": all_correct / len(complete) if complete else None,
        "trajectories": total,
        "first_actions": dict(first_actions),
        "transitions": dict(transitions),
        "behavior": dict(behavior),
        "behavior_rate": {
            key: (value / total if total else None) for key, value in behavior.items()
        },
        "note": "task-level audit over the whole run; not dev200/test1534",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--task-file", default="artifacts/datasets/rl_e1_1.parquet")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", default=None)
    a = p.parse_args()
    result = analyze(a.trace_dir, a.task_file, limit=a.limit)
    out = Path(a.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.out_md:
        lines = [
            "# RL Trace Analysis",
            "",
            f"- traces checked: {result['traces_checked']}",
            f"- complete n4 groups: {result['groups_complete']}",
            f"- mixed: {result['groups_mixed']} ({result['mixed_rate']})",
            f"- all-zero: {result['groups_all_zero']} ({result['all_zero_rate']})",
            f"- all-correct: {result['groups_all_correct']} ({result['all_correct_rate']})",
            "",
            "## First actions",
            "",
            "| action | count |",
            "|---|---:|",
        ]
        for key, value in sorted(result["first_actions"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{key}` | {value} |")
        lines += ["", "## Transitions", "", "| transition | count |", "|---|---:|"]
        for key, value in sorted(result["transitions"].items()):
            lines.append(f"| `{key}` | {value} |")
        lines += [
            "",
            "## Behavior",
            "",
            "| metric | count | rate |",
            "|---|---:|---:|",
        ]
        for key, value in sorted(result["behavior"].items()):
            lines.append(f"| `{key}` | {value} | {result['behavior_rate'].get(key)} |")
        lines += ["", result["note"], ""]
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
