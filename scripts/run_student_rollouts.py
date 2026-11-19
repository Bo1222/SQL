"""Multi-card student-only rollout runner for M6-08m P1/P2.

One parent process shards a task file across NPUs.  Each worker owns one NPU,
loads the 3B policy once, runs the requested arm, scores every final SQL with
Gold, and writes a shard JSONL.  The parent merges shards and writes a summary.

Arms:
  --mode raw --num-samples 1 --temperature 0       N=1 greedy
  --mode raw --num-samples 4 --temperature 0.8     N=4 sampling
  --mode raw --num-samples 8 --temperature 0.8     N=8 sampling
  --mode repair --repair-k 4                       N=1 + failed-prefix repair

The runner records provenance for every trajectory.  Gold is used only for
final-EX scoring; it never enters a prompt, observation or prefix.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e1_alignment.provenance import (  # noqa: E402
    RAW_FAILURE,
    RAW_SUCCESS,
    SELF_REPAIR_FAILURE,
    SELF_REPAIR_SUCCESS,
    finalize_provenance,
    provenance_for_state,
    sha256_file,
    write_json,
)

SQL_WS = re.compile(r"\s+")
_CHILDREN = []


def _stop_children(signum, frame):
    for proc in _CHILDREN:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    raise SystemExit(128 + signum)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_prefix(path):
    """Read complete JSONL records, stopping at a truncated tail after a crash."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_sql(sql):
    return SQL_WS.sub(" ", (sql or "").strip()).lower().rstrip(";").strip()


def load_fingerprint(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = data.get("model") or {}
    return {
        "path": str(path),
        "runtime_fingerprint": data.get("fingerprint_sha256"),
        "model_sha256": model.get("model_sha256"),
        "model_path": model.get("model_path"),
        "contract": data.get("contract") or {},
        "format": data.get("format"),
    }


def validate_fingerprint(fingerprint, args):
    if not (fingerprint["model_sha256"] and fingerprint["runtime_fingerprint"]
            and fingerprint["model_path"]):
        raise SystemExit("fingerprint lacks model_sha256, fingerprint_sha256 or model_path")
    expected_model = Path(fingerprint["model_path"])
    if not expected_model.is_absolute():
        expected_model = ROOT / expected_model
    if expected_model.resolve() != Path(args.model).resolve():
        raise SystemExit(
            f"--model {args.model} does not match fingerprint model {expected_model}")
    contract_version = (fingerprint["contract"] or {}).get("retrieval_version")
    if contract_version and contract_version != args.retrieval_version:
        raise SystemExit(
            f"--retrieval-version {args.retrieval_version} does not match "
            f"fingerprint contract {contract_version}")


def _load_eval_ids(path):
    ids, dbs = set(), set()
    for row in read_jsonl(path):
        if row.get("id"):
            ids.add(row["id"])
        if row.get("db_id"):
            dbs.add(row["db_id"])
    return ids, dbs


def validate_tasks(tasks, args):
    required = ("question_id", "db_id", "question", "gold_sql", "db_path", "split")
    eval_ids, eval_dbs = set(), set()
    for path in (args.dev200, args.val, args.test):
        ids, dbs = _load_eval_ids(path)
        eval_ids |= ids
        eval_dbs |= dbs
    seen = set()
    for task in tasks:
        missing = [key for key in required if not task.get(key)]
        if missing:
            raise SystemExit(f"task {task.get('question_id')} missing {missing}")
        if task["split"] not in ("sft", "sft_train"):
            raise SystemExit(f"task {task['question_id']} has non-SFT split {task['split']}")
        if task["question_id"] in seen:
            raise SystemExit(f"duplicate task {task['question_id']}")
        seen.add(task["question_id"])
        if task["question_id"] in eval_ids:
            raise SystemExit(f"task {task['question_id']} overlaps an evaluation question")
        if task["db_id"] in eval_dbs:
            raise SystemExit(f"task {task['question_id']} uses evaluation database {task['db_id']}")
        if not Path(task["db_path"]).is_file():
            raise SystemExit(f"task {task['question_id']} database missing: {task['db_path']}")


def lookup_counts(tool_events):
    empty = error = sql_error = 0
    for event in tool_events or []:
        action = event.get("action")
        result = event.get("result") or {}
        if action == "schema_lookup":
            metadata = result.get("retrieval_metadata") or {}
            tables = metadata.get("matched_tables")
            if tables is None:
                tables = result.get("tables")
            empty += int(bool(result.get("status") == "ok" and not tables))
            error += int(bool(result.get("status") != "ok"))
        elif action == "sql_execute":
            sql_error += int(bool(result.get("status") != "ok"))
    return empty, error, sql_error


def state_record(state, task, *, arm, generation_mode, sample_index,
                 parent_trajectory_id, repair_round, temperature, top_p,
                 fingerprint, retrieval_version):
    record = state.record()
    empty, lookup_error, sql_error = lookup_counts(record.get("tool_events"))
    task_split = task.get("split")
    if task_split not in ("sft", "sft_train"):
        raise ValueError(f"non-SFT task split: {task_split!r}")
    provenance = provenance_for_state(
        state=state,
        source=SELF_REPAIR_SUCCESS if repair_round else RAW_SUCCESS,
        generation_mode=generation_mode,
        split="sft_train",
        model_path=fingerprint["model_path"] or fingerprint["path"],
        model_fingerprint=fingerprint["model_sha256"] or "unknown",
        runtime_fingerprint=fingerprint["runtime_fingerprint"] or "unknown",
        retrieval_version=retrieval_version,
        prompt_mode="query",
        observation_mode="compact",
        sample_index=sample_index,
        parent_trajectory_id=parent_trajectory_id,
        repair_round=repair_round,
    )
    final_ex = False
    final_sql = record.get("final_sql")
    score = {"correct": False, "ok": False, "error": "not_scored"}
    score_error = False
    from reward.bird_ex_reward import score_sql
    for attempt in range(2):
        try:
            score = score_sql(task["db_path"], final_sql, task["gold_sql"])
            if score.get("infrastructure_error"):
                score_error = True
                continue
            score_error = False
            break
        except Exception as exc:  # scoring failure is recorded, never silently passed
            score = {"correct": False, "ok": False,
                     "error": f"{type(exc).__name__}: {exc}",
                     "infrastructure_error": True, "attempt": attempt + 1}
            score_error = True
    if score_error:
        final_ex = False
        score["score_error"] = True
    else:
        final_ex = bool(score.get("correct"))
    finalize_provenance(provenance, final_ex=final_ex, final_sql=final_sql,
                        final_ex_source="score_error" if score_error else None)
    rollout_id = f"{task['question_id']}::{arm}::s{sample_index}"
    if repair_round:
        rollout_id = f"{parent_trajectory_id}::repair::r{repair_round}::k{sample_index}"
    return {
        "rollout_id": rollout_id,
        "task_index": task.get("task_index"),
        "question_id": task["question_id"],
        "group": task.get("group"),
        "split": "sft_train",
        "db_id": task["db_id"],
        "arm": arm,
        "generation_mode": generation_mode,
        "sample_index": sample_index,
        "parent_trajectory_id": parent_trajectory_id,
        "repair_round": repair_round,
        "temperature": temperature,
        "top_p": top_p,
        "retrieval_version": retrieval_version,
        "provenance": provenance,
        "final_sql": final_sql,
        "final_ex": final_ex,
        "score": score,
        "score_error": score_error,
        "lookup_empty": empty,
        "lookup_error": lookup_error,
        "sql_error": sql_error,
        "record": record,
    }


def _max_event_index(segments):
    indexes = [segment.get("tool_event_index") for segment in segments
               if segment.get("tool_event_index") is not None]
    return max(indexes) if indexes else -1


def _prefix_segments(segments):
    # Runtime.new_state already inserts the prompt; reconstructed prefixes must
    # contain only action/observation segments.
    return [segment for segment in segments if segment.get("kind") != "prompt"]


def find_repair_prefix(state):
    """Return (prefix_segments, prefix_tool_events, reason) or (None, None, reason)."""
    segments = state.segments
    tool_events = state.tool_events
    for index, segment in enumerate(segments):
        if segment.get("kind") != "action":
            continue
        if segment.get("invalid"):
            prefix = _prefix_segments(segments[:index])
            return prefix, tool_events[: _max_event_index(prefix) + 1], "invalid_action"
        event_index = segment.get("tool_event_index")
        if event_index is None:
            continue
        event = tool_events[event_index]
        result = event.get("result") or {}
        if event.get("action") == "sql_execute" and result.get("status") != "ok":
            return _prefix_segments(segments[:index]), tool_events[:event_index], "sql_error"
        if event.get("action") == "schema_lookup":
            metadata = result.get("retrieval_metadata") or {}
            tables = metadata.get("matched_tables")
            if tables is None:
                tables = result.get("tables")
            if result.get("status") != "ok" or not tables:
                return _prefix_segments(segments[:index]), tool_events[:event_index], "lookup_empty_or_error"
    for index in range(len(segments) - 1, -1, -1):
        segment = segments[index]
        if segment.get("kind") == "action" and segment.get("action") == "answer":
            prefix = _prefix_segments(segments[:index])
            return prefix, tool_events[: _max_event_index(prefix) + 1], "wrong_answer"
    return None, None, "no_repair_point"


def run_task(task, runtime, engine, args, fingerprint):
    from e1_alignment.runtime import run_continuations, run_group

    row = {"id": task["id"], "question": task["question"],
           "evidence": task.get("evidence", ""), "gold_sql": task["gold_sql"],
           "db_id": task["db_id"], "db_path": task["db_path"]}
    results = []
    if args.mode == "raw" and args.num_samples not in (1, 4, 8):
        raise ValueError("raw mode supports num_samples in {1,4,8}")
    if args.mode == "raw":
        n = args.num_samples
        mode = ("n1_greedy" if n == 1 and args.temperature == 0
                else "n1_sample" if n == 1
                else "n4_sample" if n == 4 else "n8_sample")
        states = run_group(row, runtime, engine, n=n,
                           temperature=args.temperature, top_p=args.top_p,
                           seed_base=args.seed + task.get("task_index", 0) * 1009)
        for sample_index, state in enumerate(states):
            results.append(state_record(
                state, task, arm=args.arm, generation_mode=mode,
                sample_index=sample_index, parent_trajectory_id=None,
                repair_round=0, temperature=args.temperature, top_p=args.top_p,
                fingerprint=fingerprint, retrieval_version=args.retrieval_version))
        return results

    # repair mode: one greedy base pass, then failed-prefix continuations.
    states = run_group(row, runtime, engine, n=1, temperature=0.0, top_p=1.0,
                       seed_base=args.seed + task.get("task_index", 0) * 1009)
    base = states[0]
    base_record = state_record(
        base, task, arm=args.arm, generation_mode="n1_greedy",
        sample_index=0, parent_trajectory_id=None, repair_round=0,
        temperature=0.0, top_p=1.0, fingerprint=fingerprint,
        retrieval_version=args.retrieval_version)
    results.append(base_record)
    if base_record["final_ex"] or args.repair_k <= 0:
        return results

    prefix, tool_events, reason = find_repair_prefix(base)
    if not prefix:
        # An empty prefix means the first action itself failed.  Continuing
        # from the prompt is just a fresh sample, not a failed-prefix repair,
        # so it is deliberately not counted as self-repair evidence.
        base_record["repair"] = {
            "attempted": False,
            "reason": reason if prefix is None else "empty_prefix",
        }
        return results
    continuations = run_continuations(
        row, runtime, engine, prefix_segments=prefix,
        n=args.repair_k, temperature=args.repair_temperature,
        top_p=args.repair_top_p,
        seed_base=args.seed + task.get("task_index", 0) * 1009 + 7919,
        prefix_tool_events=tool_events)
    for index, state in enumerate(continuations):
        record = state_record(
            state, task, arm=args.arm, generation_mode="prefix_repair",
            sample_index=index, parent_trajectory_id=base_record["rollout_id"],
            repair_round=1, temperature=args.repair_temperature,
            top_p=args.repair_top_p, fingerprint=fingerprint,
            retrieval_version=args.retrieval_version)
        record["repair"] = {"reason": reason, "prefix_segments": len(prefix),
                            "prefix_tool_events": len(tool_events)}
        results.append(record)
    return results


def worker_main(args):
    # vLLM must be imported only after ASCEND_RT_VISIBLE_DEVICES is set by the parent.
    from e1_alignment.engine import VLLMBatchEngine
    from e1_alignment.runtime import Runtime, RuntimeContract

    fingerprint = load_fingerprint(args.fingerprint)
    validate_fingerprint(fingerprint, args)
    contract = RuntimeContract(
        model_path=args.model,
        retrieval_version=args.retrieval_version,
        max_rounds=args.max_rounds,
        max_tools=args.max_tools,
        max_schema_lookups=args.max_schema_lookups,
        max_sql_executes=args.max_sql_executes,
        max_context=args.max_model_len,
        max_observation_tokens=args.max_observation_tokens,
    )
    runtime = Runtime(contract)
    engine = VLLMBatchEngine(args.model, max_model_len=args.max_model_len,
                             gpu_memory_utilization=args.gpu_memory_utilization)
    tasks = [task for task in read_jsonl(args.tasks)
             if args.task_offset <= int(task.get("task_index", 0))
             and (not args.task_limit or int(task.get("task_index", 0)) < args.task_offset + args.task_limit)
             and int(task.get("task_index", 0)) % args.num_shards == args.shard_index]
    validate_tasks(tasks, args)
    shard_path = Path(args.out_dir) / f"shard_{args.shard_index}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume:
        if args.mode != "raw":
            raise SystemExit("--resume is supported only for raw mode")
        if shard_path.exists():
            existing = read_jsonl_prefix(shard_path)
            grouped = defaultdict(list)
            for record in existing:
                grouped[record.get("task_index")].append(record)
            keep = []
            expected_samples = set(range(args.num_samples))
            for task_index, records in grouped.items():
                sample_indices = {r.get("sample_index") for r in records}
                if (len(records) == args.num_samples
                        and sample_indices == expected_samples):
                    completed.add(task_index)
                    keep.extend(records)
            with shard_path.open("w", encoding="utf-8") as handle:
                for record in keep:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    started = time.monotonic()
    with shard_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for task in tasks:
            if task.get("task_index") in completed:
                continue
            task_started = time.monotonic()
            try:
                records = run_task(task, runtime, engine, args, fingerprint)
            except Exception as exc:
                import traceback
                records = [{
                    "rollout_id": f"{task['question_id']}::{args.arm}::exception",
                    "task_index": task.get("task_index"),
                    "question_id": task["question_id"],
                    "group": task.get("group"), "db_id": task["db_id"],
                    "arm": args.arm, "generation_mode": "exception",
                    "final_sql": None, "final_ex": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                    "record": {"infrastructure_error": True,
                               "finish_reason": "exception",
                               "segments": [], "tool_events": []},
                    "provenance": None,
                }]
            for record in records:
                record["wall_s"] = round(time.monotonic() - task_started, 3)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(json.dumps({"shard": args.shard_index, "task": task.get("task_index"),
                              "question_id": task["question_id"],
                              "records": len(records),
                              "ex": sum(r.get("final_ex") for r in records),
                              "wall_s": round(time.monotonic() - task_started, 2)}),
                  flush=True)
    write_json(Path(args.out_dir) / f"worker_{args.shard_index}.json",
               {"shard": args.shard_index, "tasks": len(tasks),
                "wall_s": round(time.monotonic() - started, 2)})


def summarize(out_dir, args):
    rows = []
    for path in sorted(Path(out_dir).glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda r: (r.get("task_index") or 0, r.get("arm"), r.get("sample_index") or 0,
                             r.get("rollout_id")))
    write_jsonl(Path(out_dir) / "rollouts.jsonl", rows)
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    task_wall = {}
    for row in rows:
        question = row["question_id"]
        task_wall[question] = max(task_wall.get(question, 0.0), float(row.get("wall_s") or 0.0))
    mean_task_wall = (sum(task_wall.values()) / len(task_wall)) if task_wall else 0.0
    success_questions = {q for q, items in by_question.items() if any(r.get("final_ex") for r in items)}
    success_sql = {norm_sql(r["final_sql"]) for r in rows if r.get("final_ex") and r.get("final_sql")}
    expected = {task["question_id"] for task in read_jsonl(args.tasks)
                if args.task_offset <= int(task.get("task_index", 0))
                and (not args.task_limit
                     or int(task.get("task_index", 0)) < args.task_offset + args.task_limit)}
    missing_tasks = sorted(expected - set(by_question))
    incomplete_tasks = {}
    if args.mode == "raw":
        expected_samples = set(range(args.num_samples))
        for question, items in by_question.items():
            sample_indices = {r.get("sample_index") for r in items}
            if len(items) != args.num_samples or sample_indices != expected_samples:
                incomplete_tasks[question] = {
                    "records": len(items),
                    "sample_indices": sorted(sample_indices, key=lambda x: (x is None, x)),
                }
    summary = {
        "task_file": args.tasks,
        "out_dir": str(out_dir),
        "mode": args.mode,
        "arm": args.arm,
        "retrieval_version": args.retrieval_version,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repair_k": args.repair_k if args.mode == "repair" else 0,
        "tasks": len(by_question),
        "rollouts": len(rows),
        "success_rollouts": sum(bool(r.get("final_ex")) for r in rows),
        "success_questions": len(success_questions),
        "coverage": round(len(success_questions) / max(len(by_question), 1), 4),
        "unique_success_sql": len(success_sql),
        "invalid_actions": sum(int((r.get("record") or {}).get("invalid_actions") or 0) for r in rows),
        "infrastructure_errors": sum(bool((r.get("record") or {}).get("infrastructure_error")) for r in rows),
        "score_errors": sum(bool(r.get("score_error")) for r in rows),
        "exceptions": sum(bool(r.get("exception")) for r in rows),
        "lookup_empty": sum(int(r.get("lookup_empty") or 0) for r in rows),
        "lookup_error": sum(int(r.get("lookup_error") or 0) for r in rows),
        "sql_error": sum(int(r.get("sql_error") or 0) for r in rows),
        "wall_s": round(sum(float(r.get("wall_s") or 0) for r in rows), 3),
        "mean_task_wall_s": round(mean_task_wall, 3),
        "estimated_full_tasks_4cards_h": round(
            mean_task_wall * len(expected) / 4 / 3600, 3),
        "estimated_full_4471_4cards_h": round(mean_task_wall * 4471 / 4 / 3600, 3),
        "expected_tasks": len(expected),
        "missing_tasks": missing_tasks[:50],
        "missing_task_count": len(missing_tasks),
        "incomplete_tasks": dict(list(incomplete_tasks.items())[:50]),
        "incomplete_task_count": len(incomplete_tasks),
        "groups": {},
        "provenance_counts": dict(Counter((r.get("provenance") or {}).get("source")
                                          for r in rows)),
    }
    summary["ok"] = not missing_tasks and not incomplete_tasks
    summary["clean"] = (summary["exceptions"] == 0
                        and summary["infrastructure_errors"] == 0
                        and summary["score_errors"] == 0)
    for group in sorted({r.get("group") for r in rows if r.get("group")}):
        items = [r for r in rows if r.get("group") == group]
        questions = {r["question_id"] for r in items}
        summary["groups"][group] = {
            "tasks": len(questions),
            "rollouts": len(items),
            "success_questions": len({r["question_id"] for r in items if r.get("final_ex")}),
            "coverage": round(len({r["question_id"] for r in items if r.get("final_ex")})
                              / max(len(questions), 1), 4),
            "lookup_empty": sum(int(r.get("lookup_empty") or 0) for r in items),
            "sql_error": sum(int(r.get("sql_error") or 0) for r in items),
        }
    write_json(Path(out_dir) / "summary.json", summary)
    lines = [
        f"# M6-08m rollout summary: {args.arm}", "",
        f"- tasks: **{summary['tasks']}**",
        f"- rollouts: **{summary['rollouts']}**",
        f"- success rollouts: **{summary['success_rollouts']}**",
        f"- success questions: **{summary['success_questions']} / {summary['tasks']}** "
        f"({summary['coverage']:.1%})",
        f"- unique success SQL: **{summary['unique_success_sql']}**",
        f"- invalid actions: {summary['invalid_actions']}; "
        f"infrastructure: {summary['infrastructure_errors']}; "
        f"score errors: {summary['score_errors']}; exceptions: {summary['exceptions']}",
        f"- lookup empty: {summary['lookup_empty']}; lookup error: {summary['lookup_error']}; "
        f"SQL error: {summary['sql_error']}",
        f"- wall: {summary['wall_s']:.1f} s",
        f"- expected tasks: {summary['expected_tasks']}; "
        f"missing: {summary['missing_task_count']}; "
        f"incomplete: {summary['incomplete_task_count']}; ok: **{summary['ok']}**",
        f"- provenance: `{json.dumps(summary['provenance_counts'], ensure_ascii=False)}`",
        "",
        "## By group",
        "",
        "| group | tasks | rollouts | success questions | coverage | lookup empty | SQL error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in summary["groups"].items():
        lines.append(f"| {group} | {item['tasks']} | {item['rollouts']} | "
                     f"{item['success_questions']} | {item['coverage']:.1%} | "
                     f"{item['lookup_empty']} | {item['sql_error']} |")
    (Path(out_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _clean_run_dir(out):
    for pattern in ("shard_*.jsonl", "worker_*.json", "worker_*.log",
                    "worker_failures.json"):
        for path in out.glob(pattern):
            path.unlink()
    for name in ("rollouts.jsonl", "summary.json", "summary.md", "run_config.json"):
        path = out / name
        if path.exists():
            path.unlink()


def _run_config(args, workers, fingerprint):
    return {
        "args": vars(args),
        "tasks": str(Path(args.tasks).resolve()),
        "task_file_sha256": sha256_file(args.tasks),
        "fingerprint": str(Path(args.fingerprint).resolve()),
        "fingerprint_sha256": fingerprint["runtime_fingerprint"],
        "model": str(Path(args.model).resolve()),
        "runner_sha256": sha256_file(__file__),
        "num_workers": workers,
        "num_shards": workers,
    }


def _verify_resume_config(out, current, allow_runner_change=False):
    path = out / "run_config.json"
    if not path.is_file():
        raise SystemExit(f"--resume requires {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    keys = ("tasks", "task_file_sha256", "fingerprint_sha256", "model",
            "num_shards", "num_workers")
    for key in keys:
        if previous.get(key) != current.get(key):
            raise SystemExit(
                f"--resume config mismatch for {key}: {previous.get(key)!r} != {current.get(key)!r}")
    if previous.get("runner_sha256") != current.get("runner_sha256"):
        if not allow_runner_change:
            raise SystemExit(
                "runner changed since this run started; pass --allow-runner-change "
                "only after confirming the edit cannot change raw rollout semantics")
        print(json.dumps({"resume_runner_changed": True,
                          "previous": previous.get("runner_sha256"),
                          "current": current.get("runner_sha256")}), flush=True)
    prev_args, cur_args = previous.get("args", {}), current.get("args", {})
    for key in ("mode", "arm", "num_samples", "temperature", "top_p",
                "repair_k", "repair_temperature", "repair_top_p",
                "retrieval_version", "task_offset", "task_limit",
                "max_model_len", "max_rounds", "max_tools",
                "max_schema_lookups", "max_sql_executes",
                "max_observation_tokens", "seed", "dev200", "val", "test"):
        if prev_args.get(key) != cur_args.get(key):
            raise SystemExit(
                f"--resume arg mismatch for {key}: {prev_args.get(key)!r} != {cur_args.get(key)!r}")


def parent_main(args):
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    if args.resume and args.mode != "raw":
        raise SystemExit("--resume is supported only for raw mode")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _clean_run_dir(out)
    elif args.resume:
        if not (out / "run_config.json").is_file():
            raise SystemExit(f"{out}/run_config.json missing; cannot resume")
    else:
        leftovers = list(out.glob("shard_*.jsonl")) + [
            out / name for name in ("summary.json", "run_config.json", "rollouts.jsonl")
            if (out / name).exists()]
        if leftovers:
            raise SystemExit(
                f"{out} contains a partial/finished run; use --resume or --overwrite")
    tasks = list(read_jsonl(args.tasks))
    selected = [task for task in tasks
                if args.task_offset <= int(task.get("task_index", 0))
                and (not args.task_limit or int(task.get("task_index", 0)) < args.task_offset + args.task_limit)]
    if not selected:
        raise SystemExit("no tasks in the requested offset/limit range")
    validate_tasks(selected, args)
    cards = [card.strip() for card in args.cards.split(",") if card.strip()]
    workers = min(args.num_workers, len(cards), len(selected))
    if workers < 1:
        raise SystemExit("no workers")
    fingerprint = load_fingerprint(args.fingerprint)
    validate_fingerprint(fingerprint, args)
    write_json(out / "fingerprint_used.json", fingerprint)
    run_config = _run_config(args, workers, fingerprint)
    if args.resume:
        _verify_resume_config(out, run_config, args.allow_runner_change)
    else:
        write_json(out / "run_config.json", run_config)

    signal.signal(signal.SIGINT, _stop_children)
    signal.signal(signal.SIGTERM, _stop_children)
    procs, logs = [], []
    for index in range(workers):
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker", "--tasks", args.tasks, "--out-dir", str(out),
            "--shard-index", str(index), "--num-shards", str(workers),
            "--model", args.model, "--fingerprint", args.fingerprint,
            "--mode", args.mode, "--arm", args.arm,
            "--num-samples", str(args.num_samples),
            "--temperature", str(args.temperature), "--top-p", str(args.top_p),
            "--repair-k", str(args.repair_k),
            "--repair-temperature", str(args.repair_temperature),
            "--repair-top-p", str(args.repair_top_p),
            "--retrieval-version", args.retrieval_version,
            "--max-model-len", str(args.max_model_len),
            "--max-rounds", str(args.max_rounds),
            "--max-tools", str(args.max_tools),
            "--max-schema-lookups", str(args.max_schema_lookups),
            "--max-sql-executes", str(args.max_sql_executes),
            "--max-observation-tokens", str(args.max_observation_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--seed", str(args.seed),
            "--task-offset", str(args.task_offset),
            "--task-limit", str(args.task_limit),
            "--dev200", args.dev200,
            "--val", args.val,
            "--test", args.test,
        ]
        if args.resume:
            command.append("--resume")
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = cards[index]
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.setdefault("OMP_NUM_THREADS", "4")
        log = (out / f"worker_{index}.log").open(
            "a" if args.resume else "w", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        procs.append(proc)
        _CHILDREN.append(proc)
        logs.append(log)
        print(json.dumps({"launched": index, "card": cards[index],
                          "pid": proc.pid}), flush=True)
    failed = []
    run_started = time.monotonic()
    for index, proc in enumerate(procs):
        timeout = None
        if args.max_run_seconds > 0:
            timeout = max(1.0, args.max_run_seconds - (time.monotonic() - run_started))
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for child in procs:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            failed.append({"worker": index, "error": "max_run_seconds_exceeded"})
            break
        if code != 0:
            failed.append({"worker": index, "exit_code": code})
    for log in logs:
        log.close()
    if failed:
        write_json(out / "worker_failures.json", failed)
        raise SystemExit(f"worker failures: {failed}")
    summary = summarize(out, args)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not summary["ok"]:
        raise SystemExit(
            f"run incomplete: missing={summary['missing_task_count']}, "
            f"incomplete={summary['incomplete_task_count']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    parser.add_argument("--fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    parser.add_argument("--mode", choices=("raw", "repair"), default="raw")
    parser.add_argument("--arm", default="n1")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repair-k", type=int, default=4)
    parser.add_argument("--repair-temperature", type=float, default=0.8)
    parser.add_argument("--repair-top-p", type=float, default=0.95)
    parser.add_argument("--retrieval-version", default="E1", choices=("E1", "E1.1"))
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-tools", type=int, default=4)
    parser.add_argument("--max-schema-lookups", type=int, default=2)
    parser.add_argument("--max-sql-executes", type=int, default=2)
    parser.add_argument("--max-observation-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cards", default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0,
                        help="0 means no limit; useful for calibration slices")
    parser.add_argument("--dev200", default="artifacts/datasets/dev200.jsonl")
    parser.add_argument("--val", default="data/processed/bird_agent/val.jsonl")
    parser.add_argument("--test", default="data/processed/bird_agent/test.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip already-complete raw-mode tasks in existing shards")
    parser.add_argument("--allow-runner-change", action="store_true",
                        help="resume despite a runner source hash change; use only after "
                             "confirming raw-mode semantics are unchanged")
    parser.add_argument("--max-run-seconds", type=float, default=0,
                        help="0 disables; otherwise kill workers after this wall time")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
