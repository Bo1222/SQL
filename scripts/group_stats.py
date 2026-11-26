"""M6-04: per-group reward composition from a training rollout trace.

The trainer logs only batch-level scores, so group composition (mixed vs all-zero
vs all-correct) is reconstructed by scoring each traced rollout against its gold
SQL and grouping by (step, sample_index).

Also reports coverage/protocol health so one command answers "is the signal usable".
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402


def read_records(path):
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


def classify(group_rewards):
    total = sum(group_rewards)
    if total == 0:
        return "all_zero"
    if total == len(group_rewards):
        return "all_correct"
    return "mixed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="reports/m6/traces_pilot")
    p.add_argument("--data", default="data/processed/bird_agent/rl.jsonl")
    p.add_argument("--out", default="reports/m6/group_signal.json")
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--timeout", type=int, default=30)
    a = p.parse_args()

    records = read_records(a.trace)
    if not records:
        raise SystemExit(f"no rollout records at {a.trace}")
    gold = {row["id"]: row for row in (json.loads(line) for line in
                                       Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip())}

    # score every rollout once
    for record in records:
        gold_row = gold.get(record.get("question_id"))
        if not gold_row:
            record["_reward"] = 0.0
            record["_score_error"] = "missing_gold"
            continue
        scored = score_sql(gold_row["db_path"], record.get("final_sql"), gold_row["gold_sql"], timeout=a.timeout)
        record["_reward"] = 1.0 if scored.get("correct") else 0.0
        record["_score_error"] = scored.get("error")

    groups = defaultdict(list)
    for record in records:
        groups[(record.get("step"), record.get("question_id"))].append(record)
    # a partial step (killed run) leaves groups with fewer than n rollouts; drop them
    expected = max(Counter(len(v) for v in groups.values()).items(), key=lambda kv: (kv[1], kv[0]))[0]
    dropped = [key for key, value in groups.items() if len(value) != expected]
    for key in dropped:
        groups.pop(key)
    # drop whole partial steps (a killed run leaves a step with fewer groups than the mode)
    per_step_counts = Counter(step for step, _ in groups)
    if per_step_counts:
        full_size = max(per_step_counts.values())
        partial_steps = sorted(step for step, count in per_step_counts.items() if count != full_size)
        groups = {key: value for key, value in groups.items() if key[0] not in set(partial_steps)}
    else:
        partial_steps = []

    steps = sorted({step for step, _ in groups if step is not None})
    per_step = []
    for step in steps:
        batch_keys = [key for key in groups if key[0] == step]
        counts = Counter()
        rewards = []
        for key in batch_keys:
            values = [r["_reward"] for r in groups[key]]
            counts[classify(values)] += 1
            rewards.extend(values)
        total_groups = sum(counts.values())
        per_step.append({
            "step": step,
            "groups": total_groups,
            "mixed": counts.get("mixed", 0),
            "all_zero": counts.get("all_zero", 0),
            "all_correct": counts.get("all_correct", 0),
            "mixed_rate": round(counts.get("mixed", 0) / max(total_groups, 1), 3),
            "reward_mean": round(sum(rewards) / max(len(rewards), 1), 3),
            "protocol_errors": sum(1 for r in records if r.get("step") == step and r.get("finish_reason") == "invalid_action"),
        })

    steady = [row for row in per_step if row["step"] > a.warmup_steps] or per_step
    total_groups = sum(row["groups"] for row in steady)
    total_mixed = sum(row["mixed"] for row in steady)
    summary = {
        "steps": len(steady),
        "warmup_steps_excluded": a.warmup_steps,
        "groups": total_groups,
        "mixed": total_mixed,
        "all_zero": sum(row["all_zero"] for row in steady),
        "all_correct": sum(row["all_correct"] for row in steady),
        "mixed_rate": round(total_mixed / max(total_groups, 1), 3),
        "reward_mean": round(sum(row["reward_mean"] for row in steady) / max(len(steady), 1), 3),
        "protocol_errors": sum(row["protocol_errors"] for row in steady),
        "effective_pg_groups_per_step": round(total_mixed / max(len(steady), 1), 2),
    }

    thresholds = {
        "mixed_rate_ge_50": summary["mixed_rate"] >= 0.5,
        "mixed_rate_25_to_50": 0.25 <= summary["mixed_rate"] < 0.5,
        "mixed_rate_lt_25": summary["mixed_rate"] < 0.25,
    }
    if thresholds["mixed_rate_ge_50"]:
        action = "keep n=4 and 8 prompts"
    elif thresholds["mixed_rate_25_to_50"]:
        action = "raise prompts/update to 16 before increasing n"
    else:
        action = "implement bounded dynamic sampling (filter + resample) and A/B it"
    summary["decision"] = {"rule": "TASKS_M6 M6-04", **thresholds, "action": action,
                           "enough_signal_for_pilot_gate": summary["mixed_rate"] >= 0.25}

    summary["groups_dropped"] = len(dropped)
    summary["partial_steps_dropped"] = sorted(partial_steps)
    report = {"format": "m6-04-group-signal-v1", "trace": str(a.trace), "summary": summary,
              "per_step": per_step,
              "note": "groups keyed by (step, question_id); incomplete groups (partial/killed steps) dropped; expected group size inferred from the mode"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({"per_step": per_step}, ensure_ascii=False))


if __name__ == "__main__":
    main()




