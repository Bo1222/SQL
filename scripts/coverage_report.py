"""M6-01: throughput calibration + prompt/DB coverage from the rollout trace.

Produces
  reports/m6/coverage.json                 per-step and cumulative coverage (+ health/timing)
  reports/m6/throughput_calibration.json   step timing, stage shares, token accounting
  <log-dir>/coverage_log.jsonl             one line per step, kept next to the training log

A run always burns its first step on the dummy-weight rollout (veRL loads the
rollout engine with `load_format=dummy` and the first sync only lands at the end
of step 1), so the report separates the warm-up step from steady-state numbers.
"""
import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

TIMING_KEYS = ["timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor",
               "timing_s/update_weights", "timing_s/step", "perf/total_num_tokens",
               "perf/max_memory_allocated_gb", "response_length/mean", "prompt_length/mean",
               "num_turns/mean", "critic/score/mean", "actor/kl_loss", "actor/grad_norm"]


def read_records(path):
    """Accept a directory of rollout shards, a single file, or a glob."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("rollouts*.jsonl"))
    elif path.is_file():
        files = [path]
    else:
        files = sorted(Path().glob(str(path)))
    rows = []
    for file in files:
        rows.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def parse_train_log(path):
    """Extract per-step scalar metrics from the veRL console log."""
    path = Path(path)
    if not path.is_file():
        return {}
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pair = re.compile(r"([A-Za-z0-9_/\.]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)")
    steps = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ansi.sub("", line)
        if "step:" not in line or "global_seqlen/min" not in line:
            continue
        metrics = {}
        for key, value in pair.findall(line):
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        if "training/global_step" in metrics:
            steps[int(metrics["training/global_step"])] = metrics
    return steps


def _step_rows(records, train: dict):
    steps = sorted({r.get("step") for r in records if r.get("step") is not None})
    cumulative_ids, cumulative_dbs, cumulative_prompt_draws = set(), set(), 0
    rows = []
    for step in steps:
        batch = [r for r in records if r.get("step") == step]
        ids = {r.get("question_id") for r in batch}
        dbs = {r.get("db_id") for r in batch}
        new_ids = ids - cumulative_ids
        cumulative_prompt_draws += len(ids)
        cumulative_ids |= ids
        cumulative_dbs |= dbs
        metrics = train.get(step, {})
        rows.append({
            "step": step,
            "trajectories": len(batch),
            "unique_prompts": len(ids),
            "new_prompts": len(new_ids),
            "rollout_duplication": round(len(batch) / max(len(ids), 1), 2),
            "cumulative_prompt_draws": cumulative_prompt_draws,
            "cumulative_unique": len(cumulative_ids),
            "cumulative_prompt_repeat_rate": round(1 - len(cumulative_ids) / max(cumulative_prompt_draws, 1), 4),
            "dbs_in_step": len(dbs),
            "cumulative_dbs": len(cumulative_dbs),
            "mean_prompt_tokens": round(st.mean(r.get("prompt_tokens", 0) for r in batch), 1),
            "mean_action_tokens": round(st.mean(r.get("action_tokens", 0) for r in batch), 1),
            "mean_observation_tokens": round(st.mean(r.get("observation_tokens", 0) for r in batch), 1),
            "mean_turns": round(st.mean(r.get("turns", 0) for r in batch), 2),
            "mean_tool_calls": round(st.mean(r.get("tool_calls", 0) for r in batch), 2),
            "lookup_rate": round(sum(1 for r in batch if r.get("schema_lookups", 0) > 0) / len(batch), 3),
            "invalid_actions": sum(1 for r in batch if r.get("finish_reason") == "invalid_action"),
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in batch)),
            "reward_mean": metrics.get("critic/score/mean"),
            "step_time_s": metrics.get("timing_s/step"),
            "gen_s": metrics.get("timing_s/gen"),
            "update_actor_s": metrics.get("timing_s/update_actor"),
            "kl": metrics.get("actor/kl_loss"),
            "grad_norm": metrics.get("actor/grad_norm"),
        })
    return rows


def _aggregate(records, rows, warmup):
    usable = [r for r in records if (r.get("step") or 0) > warmup]
    usable_rows = [row for row in rows if row["step"] > warmup]
    base_rows = usable_rows or rows
    base_records = usable or records
    prompt_draws = sum(row["unique_prompts"] for row in base_rows)
    unique_prompts = len({r.get("question_id") for r in base_records})
    times = [row["step_time_s"] for row in base_rows if row.get("step_time_s")]
    summary = {
        "steps": len(base_rows),
        "warmup_steps_excluded": warmup,
        "trajectories": len(base_records),
        "prompt_draws": prompt_draws,
        "unique_prompts": unique_prompts,
        "prompt_repeat_rate": round(1 - unique_prompts / max(prompt_draws, 1), 4),
        "rollout_duplication": round(len(base_records) / max(prompt_draws, 1), 2),
        "dbs": len({r.get("db_id") for r in base_records}),
        "lookup_rate": round(sum(1 for r in base_records if r.get("schema_lookups", 0) > 0) / max(len(base_records), 1), 3),
        "protocol_errors": sum(1 for r in base_records if r.get("finish_reason") == "invalid_action"),
        "mean_action_tokens": round(st.mean(r.get("action_tokens", 0) for r in base_records), 1),
        "mean_observation_tokens": round(st.mean(r.get("observation_tokens", 0) for r in base_records), 1),
        "mean_prompt_tokens": round(st.mean(r.get("prompt_tokens", 0) for r in base_records), 1),
    }
    if times:
        summary["step_time_s"] = {"mean": round(st.mean(times), 2), "min": round(min(times), 2),
                                  "max": round(max(times), 2),
                                  "stdev": round(st.stdev(times), 2) if len(times) > 1 else 0.0}
    rewards = [row["reward_mean"] for row in base_rows if row.get("reward_mean") is not None]
    if rewards:
        summary["reward_mean"] = round(st.mean(rewards), 3)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="reports/m6/traces_calib")
    p.add_argument("--train-log", default="logs/m6_calib.log")
    p.add_argument("--out", default="reports/m6/coverage.json")
    p.add_argument("--calibration", default="reports/m6/throughput_calibration.json")
    p.add_argument("--log-dir", default="artifacts/checkpoints/m6_calib")
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--expected-dbs", type=int, default=61)
    a = p.parse_args()

    records = read_records(a.trace)
    if not records:
        raise SystemExit(f"no rollout records at {a.trace}")
    train = parse_train_log(a.train_log)
    rows = _step_rows(records, train)

    all_steps = _aggregate(records, rows, warmup=0)
    steady = _aggregate(records, rows, warmup=a.warmup_steps)
    steady["expected_dbs"] = a.expected_dbs
    steady["db_coverage_ratio"] = round(steady["dbs"] / max(a.expected_dbs, 1), 3)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary_after_warmup": steady, "summary_all_steps": all_steps,
                               "warmup_steps": a.warmup_steps, "per_step": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "coverage_log.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"event": "coverage", **row}, ensure_ascii=False) + "\n")

    steady_steps = [train[s] for s in sorted(train) if s > a.warmup_steps] or [train[s] for s in sorted(train)]
    steps = steady_steps
    calibration = {
        "trace": str(a.trace), "train_log": str(a.train_log),
        "per_step": rows,
        "steady_state": steady,
        "token_accounting": {
            "trainer_total_num_tokens_mean": round(st.mean(s.get("perf/total_num_tokens", 0) for s in steps), 1) if steps else None,
            "trainer_prompt_tokens_mean": round(st.mean(s.get("prompt_length/mean", 0) for s in steps), 1) if steps else None,
            "trainer_response_tokens_mean": round(st.mean(s.get("response_length/mean", 0) for s in steps), 1) if steps else None,
            "trace_prompt_tokens_mean": steady["mean_prompt_tokens"],
            "trace_action_tokens_mean": steady["mean_action_tokens"],
            "trace_observation_tokens_mean": steady["mean_observation_tokens"],
            "note": "trainer total_num_tokens = per-sample (prompt + response); the trace splits action vs observation tokens",
        },
    }
    if steps:
        for key, label in (("timing_s/gen", "gen_s"), ("timing_s/old_log_prob", "old_log_prob_s"),
                           ("timing_s/ref", "ref_s"), ("timing_s/update_actor", "update_actor_s"),
                           ("timing_s/update_weights", "update_weights_s")):
            values = [s[key] for s in steps if s.get(key) is not None]
            if values:
                calibration[label] = round(st.mean(values), 2)
        mean_step = steady.get("step_time_s", {}).get("mean") or 0
        calibration["share"] = {k: round(v / mean_step, 3) for k, v in calibration.items()
                                if k.endswith("_s") and isinstance(v, (int, float)) and mean_step}
        per_step_minutes = mean_step / 60
        calibration["projection"] = {
            "step_minutes": round(per_step_minutes, 2),
            "hours_50_steps": round(mean_step * 50 / 3600, 2) if mean_step else None,
            "hours_100_steps": round(mean_step * 100 / 3600, 2) if mean_step else None,
            "hours_200_steps": round(mean_step * 200 / 3600, 2) if mean_step else None,
        }
    Path(a.calibration).parent.mkdir(parents=True, exist_ok=True)
    Path(a.calibration).write_text(json.dumps(calibration, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({"steady_state": steady, "warmup_steps": a.warmup_steps,
                      "share": calibration.get("share"), "projection": calibration.get("projection"),
                      "per_step_health": [{"step": r["step"], "invalid": r["invalid_actions"],
                                           "lookup": r["lookup_rate"], "reward": r["reward_mean"],
                                           "step_s": r["step_time_s"]} for r in rows]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

