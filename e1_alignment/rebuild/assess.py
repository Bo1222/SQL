"""Fixed-denominator monitor, task and real-prefix recovery gates."""
import argparse
from collections import Counter
import json
from pathlib import Path

from agent.protocol import parse_action
from e1_alignment.common import read_jsonl, write_json
from e1_alignment.rebuild.ast_check import extract, sqlite_schema
from e1_alignment.rebuild.visibility import coverage
from eval.paired_statistics import mcnemar
from reward.bird_ex_reward import score_sql


def validate_complete(records, fixed_rows, key="id"):
    ids = [r[key] for r in records]
    expected = [r[key] for r in fixed_rows]
    if len(set(ids)) != len(ids) or len(set(expected)) != len(expected) or set(ids) != set(expected):
        raise ValueError("evaluation_incomplete_or_not_fixed_set")
    index = {r[key]: r for r in fixed_rows}
    for r in records:
        ref = index[r[key]]
        for field in ("db_id", "question", "gold_sql"):
            if field in ref and r.get(field) != ref[field]:
                raise ValueError("evaluation_task_mismatch:" + field)


def terminal_column_errors(records):
    return sum(r.get("error_type", r.get("final_error_type")) == "no_such_column" for r in records)


def correct_count(records):
    return sum(bool(r.get("correct", r.get("final_correct"))) for r in records)


def monitor_gate(control, candidate, fixed_rows):
    validate_complete(control, fixed_rows); validate_complete(candidate, fixed_rows)
    if len(fixed_rows) != 64:
        raise ValueError("monitor_requires_fixed_64")
    correct_delta = correct_count(candidate) - correct_count(control)
    column_delta = terminal_column_errors(candidate) - terminal_column_errors(control)
    infrastructure = any(r.get("infrastructure_error") or r.get("final_infrastructure_error") for r in control + candidate)
    return {"complete": True, "n": len(fixed_rows), "control_correct": correct_count(control),
            "candidate_correct": correct_count(candidate), "correct_delta": correct_delta,
            "control_terminal_column_errors": terminal_column_errors(control),
            "candidate_terminal_column_errors": terminal_column_errors(candidate),
            "terminal_column_error_delta": column_delta,
            "pass": correct_delta > -4 and column_delta < 4 and not infrastructure,
            "infrastructure_error": infrastructure,
            "rule": "stop if correct decreases by >=4 or terminal column errors increase by >=4"}


def first_candidate(record):
    for segment in record.get("segments", []):
        if segment.get("kind") != "action":
            continue
        try:
            action = parse_action(segment.get("text", ""))
        except (ValueError, TypeError):
            continue
        if action.name in ("sql_execute", "answer"):
            return action.name, action.arguments
    # No fallback to first execution: that would lose direct-answer cases.
    return None, None


def task_metrics(records):
    first_correct, candidates, executed, grounded, grounded_den = 0, 0, 0, 0, 0
    execution_errors = Counter()
    for record in records:
        kind, sql = first_candidate(record)
        candidates += int(sql is not None)
        executed += int(any(e.get("action") == "sql_execute" for e in record.get("tool_events", [])))
        if sql:
            result = score_sql(record["db_path"], sql, record["gold_sql"])
            first_correct += int(result.get("correct", False) and not result.get("infrastructure_error"))
        for event in record.get("tool_events", []):
            if event.get("action") == "sql_execute" and event.get("result", {}).get("status") != "ok":
                execution_errors[event.get("result", {}).get("error_type", "unknown")] += 1
        # Measure only the context before the final SQL action. No metadata union.
        segments = record.get("segments", [])
        before = next((i for i in range(len(segments) - 1, -1, -1)
                       if segments[i].get("kind") == "action" and "<answer>" in segments[i].get("text", "")), len(segments))
        if record.get("final_sql"):
            grounded_den += 1
            schema = sqlite_schema(record["db_path"])
            req = extract(record["final_sql"], schema=schema)
            grounded += int(coverage(req, segments, before, schema)["ok"])
    n = len(records)
    return {"n": n, "correct": correct_count(records), "terminal_ex": correct_count(records) / max(n, 1),
            "first_candidate_correct": first_correct, "first_candidate_denominator": n,
            "first_candidate_ex": first_correct / max(n, 1), "first_candidate_present": candidates,
            "actually_executed_episodes": executed, "execution_coverage_denominator": n,
            "terminal_column_errors": terminal_column_errors(records),
            "tool_execution_errors": dict(execution_errors),
            "executable_semantic_wrong": sum(bool(r.get("valid", r.get("final_valid"))) and not bool(
                r.get("correct", r.get("final_correct"))) for r in records),
            "visible_grounded_final_sql": grounded, "visible_grounding_denominator": grounded_den,
            "grounding_rule": "bound dependencies in actual observations before final answer; unresolved AST fails"}


def paired_gate(control, candidate, fixed_rows, recovery, monitor):
    validate_complete(control, fixed_rows); validate_complete(candidate, fixed_rows)
    a, b = {r["id"]: r for r in control}, {r["id"]: r for r in candidate}
    gain = [i for i in a if not a[i].get("correct") and b[i].get("correct")]
    loss = [i for i in a if a[i].get("correct") and not b[i].get("correct")]
    p = mcnemar(len(gain), len(loss))
    def outcome(r):
        if r.get("correct"):
            return "correct"
        if r.get("valid"):
            return "semantic_wrong"
        return "terminal_" + str(r.get("error_type") or "missing_answer")
    transitions = Counter(f"{outcome(a[i])}->{outcome(b[i])}" for i in a)
    column_ok = terminal_column_errors(candidate) <= terminal_column_errors(control)
    infra = any(r.get("infrastructure_error") for r in control + candidate)
    passed = (len(gain) > len(loss) and p < .1 and column_ok and recovery.get("pass") is True
              and recovery.get("complete") is True and monitor.get("pass") is True and not infra)
    return {"complete": True, "n": len(fixed_rows), "gain": len(gain), "loss": len(loss),
            "gain_ids": gain, "loss_ids": loss, "mcnemar_exact_p": p,
            "transitions": dict(transitions), "terminal_columns_not_increased": column_ok,
            "fixed_prefix_recovery": recovery, "monitor": monitor, "pass": bool(passed),
            "rl_authorized": False, "infrastructure_error": infra,
            "control": task_metrics(control), "candidate": task_metrics(candidate)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["monitor", "dev200"], required=True)
    p.add_argument("--control", required=True); p.add_argument("--candidate", required=True)
    p.add_argument("--fixed", required=True); p.add_argument("--out", required=True)
    p.add_argument("--monitor-report"); p.add_argument("--recovery-report")
    a = p.parse_args()
    control, candidate, fixed = read_jsonl(a.control), read_jsonl(a.candidate), read_jsonl(a.fixed)
    if a.mode == "monitor":
        report = monitor_gate(control, candidate, fixed)
    else:
        if len(fixed) != 200 or not a.monitor_report or not a.recovery_report:
            raise SystemExit("complete fixed dev200 and prerequisite reports required")
        report = paired_gate(control, candidate, fixed, json.loads(Path(a.recovery_report).read_text()),
                             json.loads(Path(a.monitor_report).read_text()))
    write_json(a.out, report)
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
