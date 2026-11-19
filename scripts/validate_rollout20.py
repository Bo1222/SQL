"""Validate the 20-trajectory pilot for protocol/tool consistency and quality."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def budget_flags(segments):
    """Mark action segments whose following observation is a budget error.

    Budget-blocked actions are real protocol actions but do not call a tool, so
    tool_events legitimately has one fewer entry.  They must be excluded before
    comparing the tool-action sequence with tool_events.
    """
    flags = []
    current = None
    for s in segments or []:
        if s.get("kind") == "action":
            flags.append(False)
            current = len(flags) - 1
        elif s.get("kind") == "observation" and current is not None:
            text = str(s.get("text") or "")
            if '"error_type": "budget"' in text or "'error_type': 'budget'" in text:
                flags[current] = True
    return flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="reports/m6/e1_contract_rebuild/rollout20/rollout20.jsonl")
    p.add_argument("--out", default="reports/m6/e1_contract_rebuild/rollout20/validation.md")
    p.add_argument("--json-out", default="reports/m6/e1_contract_rebuild/rollout20/validation.json")
    a = p.parse_args()
    rows = list(read_jsonl(a.data))
    checks = Counter()
    issues = []
    for r in rows:
        new = r["new"]
        tags = new.get("action_tags") or []
        tools = new.get("tool_events") or []
        flags = budget_flags(new.get("segments") or [])
        if len(flags) != len(tags):
            issues.append({"trajectory_id": r["trajectory_id"], "kind": "budget_flag_length",
                           "actions": len(tags), "flags": len(flags)})
            flags = [False] * len(tags)
        tool_pairs = [(t, b) for t, b in zip(tags, flags) if t in ("schema_lookup", "sql_execute")]
        tool_actions = [t for t, _ in tool_pairs]
        tool_budget = [b for _, b in tool_pairs]
        expected_events = [t for t, b in tool_pairs if not b]
        event_actions = [e.get("action") for e in tools]
        checks["rows"] += 1
        checks["budget_actions"] += sum(tool_budget)
        checks["exceptions"] += int(bool(r["timing"].get("error")))
        checks["score_errors"] += int(bool(r["timing"].get("score_error")))
        checks["infrastructure"] += int(bool(new.get("infrastructure_error")))
        checks["no_answer"] += int(not new.get("final_sql"))
        checks["invalid_actions"] += int(new.get("invalid_actions") or 0)
        checks["tool_count_equal"] += int(len(tool_actions) == len(tools) + sum(tool_budget))
        checks["tool_sequence_match"] += int(expected_events == event_actions)
        checks["n_tools_equal"] += int(len(tools) == int(new.get("n_tools") or 0))
        checks["answer_tag_consistent"] += int(
            ("answer" in tags) == bool(new.get("final_sql")))
        checks["old_ex_expected"] += int(bool(r["old"].get("final_ex_expected")))
        checks["old_ex_scored"] += int(bool(r["old"].get("final_ex")))
        checks["new_ex"] += int(bool(r["new"].get("final_ex")))
        checks["tag_sequence_match"] += int(bool(r["compare"].get("tag_sequence_match")))
        checks["first_tag_match"] += int(bool(r["compare"].get("first_tag_match")))
        checks["action_sequence_exact"] += int(bool(r["compare"].get("action_sequence_exact")))
        checks["final_sql_normalized"] += int(bool(r["compare"].get("final_sql_normalized")))
        if len(tool_actions) != len(tools) + sum(tool_budget) or expected_events != event_actions:
            issues.append({"trajectory_id": r["trajectory_id"], "tags": tags,
                           "tool_actions": tool_actions, "budget": tool_budget,
                           "expected_event_actions": expected_events,
                           "event_actions": event_actions})
    n = len(rows)
    times = sorted(r["timing"]["run_s"] for r in rows)
    summary = {
        "n": n,
        "groups": dict(Counter(r["group"] for r in rows)),
        "checks": dict(checks),
        "new_ex_rate": round(checks["new_ex"] / max(n, 1), 3),
        "old_ex_rate": round(checks["old_ex_scored"] / max(n, 1), 3),
        "time_mean_s": round(sum(times) / max(n, 1), 2),
        "time_p50_s": round(times[n // 2], 2) if n else None,
        "time_max_s": round(max(times), 2) if n else None,
        "estimated_full_4471_h": round(sum(times) / max(n, 1) * 4471 / 3600, 2),
        "issues": issues[:20],
    }
    Path(a.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 20 条 policy rollout 验证", ""]
    lines.append(f"- n={n}；group={summary['groups']}")
    lines.append(f"- old EX {checks['old_ex_scored']}/{n}；new EX {checks['new_ex']}/{n}（{summary['new_ex_rate']:.1%}）")
    lines.append(f"- exceptions {checks['exceptions']}；score errors {checks['score_errors']}；"
                 f"infrastructure {checks['infrastructure']}；no answer {checks['no_answer']}；"
                 f"invalid actions {checks['invalid_actions']}")
    lines.append(f"- budget-blocked tool actions {checks['budget_actions']}（真实 action，但不调用 tool，故无 tool_event）")
    lines.append(f"- tool count equal {checks['tool_count_equal']}/{n}；tool sequence match "
                 f"{checks['tool_sequence_match']}/{n}；n_tools equal {checks['n_tools_equal']}/{n}；"
                 f"answer tag consistent {checks['answer_tag_consistent']}/{n}")
    lines.append(f"- tag sequence match {checks['tag_sequence_match']}/{n}；first tag match "
                 f"{checks['first_tag_match']}/{n}；action exact {checks['action_sequence_exact']}/{n}；"
                 f"final SQL normalized {checks['final_sql_normalized']}/{n}")
    lines.append(f"- time mean {summary['time_mean_s']:.1f}s / p50 {summary['time_p50_s']:.1f}s / max {summary['time_max_s']:.1f}s；"
                 f"mean × 4,471 = **{summary['estimated_full_4471_h']:.1f} h**（外推，单卡串行；batch-20 实测见 `../rollout20_batch/summary.md`）")
    if issues:
        lines.append("")
        lines.append("## Tool/action issues")
        for x in issues[:10]:
            lines.append(f"- {x}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
