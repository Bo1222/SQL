"""External stop monitor for Phase 1A self-sampling.

`eval_rl_learning_pool.py` has no early-stop hook, so this monitor tails its
incremental records.jsonl and kills the sampler's process group when a stop
rule fires.

Stop rules:
  * infrastructure/unavailable rewards > max_infra
  * after the first 8 seen questions have all 32 samples, if none is correct
  * at completion, if fewer than min_correct_total of the 11 seen questions have
    any correct trajectory (low yield; the wrapper must not export/continue)

The monitor is read-only over the records and writes only status artifacts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def read_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially written final line is expected while sampling.
                continue
    return rows


def summarize(rows):
    by_question = {}
    for row in rows:
        qid = row.get("id")
        if not qid:
            continue
        entry = by_question.setdefault(qid, {
            "samples": 0, "correct": 0, "unavailable": 0, "infra": 0,
            "task_index": row.get("task_index")})
        entry["samples"] += 1
        entry["correct"] += bool(row.get("correct"))
        entry["unavailable"] += not bool(row.get("reward_available", True))
        entry["infra"] += bool(row.get("infrastructure_error"))
    return by_question


def evaluate(by_question, args):
    unavailable = sum(row["unavailable"] for row in by_question.values())
    infra = sum(row["infra"] for row in by_question.values())
    complete = {qid: row for qid, row in by_question.items()
                if row["samples"] >= args.samples}
    solved = sum(1 for row in complete.values() if row["correct"] > 0)
    if unavailable > args.max_infra:
        return f"unavailable rewards {unavailable} > {args.max_infra}", complete, solved
    if infra > args.max_infra:
        return f"infrastructure errors {infra} > {args.max_infra}", complete, solved
    early = [row for row in by_question.values()
             if row.get("task_index") is not None
             and row["task_index"] < args.seen_questions
             and row["samples"] >= args.samples]
    if len(early) >= args.seen_questions:
        if sum(row["correct"] for row in early) == 0:
            return (f"first {args.seen_questions} task-index questions have zero correct "
                    "trajectories"), complete, solved
    if len(complete) >= args.total_questions and solved < args.min_correct_total:
        return (f"low yield: {solved}/{args.total_questions} questions solved "
                f"< {args.min_correct_total}"), complete, solved
    return None, complete, solved


def exit_code_for(reason):
    """3 = low yield (no kill needed), 2 = hard stop (sampler killed)."""
    return 3 if reason and reason.startswith("low yield") else 2


def kill_group(pgid, grace_s):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--seen-questions", type=int, default=8)
    parser.add_argument("--total-questions", type=int, default=11)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--min-correct-total", type=int, default=4)
    parser.add_argument("--max-infra", type=int, default=2)
    parser.add_argument("--stall-seconds", type=float, default=900.0,
                        help="kill if no new records for this long while the sampler lives")
    parser.add_argument("--kill-grace-s", type=float, default=20.0)
    args = parser.parse_args()
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    status_path = report / "monitor_status.json"
    stop_path = report / "stop_trigger.json"
    low_yield_path = report / "low_yield.json"
    state = {"last_signal": None, "last_count": None, "last_change": time.monotonic()}

    while True:
        by_question = summarize(read_records(args.records))
        reason, complete, solved = evaluate(by_question, args)
        count = sum(row["samples"] for row in by_question.values())
        if count != state["last_count"]:
            state["last_count"] = count
            state["last_change"] = time.monotonic()
        elif (not reason
              and time.monotonic() - state["last_change"] > args.stall_seconds):
            reason = f"no new records for {args.stall_seconds:g}s"
        signal_key = (sum(row["samples"] for row in by_question.values()),
                      sum(row["correct"] for row in by_question.values()),
                      len(complete), solved, reason)
        if signal_key != state["last_signal"]:
            state["last_signal"] = signal_key
            status_path.write_text(json.dumps({
                "monitor": "running",
                "questions_started": len(by_question),
                "questions_complete": len(complete),
                "questions_solved": solved,
                "records": sum(row["samples"] for row in by_question.values()),
                "reason": reason,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2) + "\n")
        if reason:
            payload = {"stopped": True, "reason": reason,
                       "questions_complete": len(complete), "questions_solved": solved,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if reason.startswith("low yield"):
                low_yield_path.write_text(json.dumps(payload, indent=2) + "\n")
            else:
                stop_path.write_text(json.dumps(payload, indent=2) + "\n")
                kill_group(args.pgid, args.kill_grace_s)
            return exit_code_for(reason)
        try:
            os.killpg(args.pgid, 0)
        except ProcessLookupError:
            # Final evaluation after the sampler exits.
            by_question = summarize(read_records(args.records))
            reason, complete, solved = evaluate(by_question, args)
            if reason:
                payload = {"stopped": True, "reason": reason,
                           "questions_complete": len(complete), "questions_solved": solved,
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                if reason.startswith("low yield"):
                    low_yield_path.write_text(json.dumps(payload, indent=2) + "\n")
                else:
                    stop_path.write_text(json.dumps(payload, indent=2) + "\n")
                return exit_code_for(reason)
            status_path.write_text(json.dumps(
                {"monitor": "exited", "questions_complete": len(complete),
                 "questions_solved": solved, "reason": "process group gone"},
                indent=2) + "\n")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
