"""Estimate large-scale training cost from measured step timings.

Inputs
  --m0   reports/m0/m0_environment_validation.json   (single-step timings, 4 prompts x n=2)
  --episodes  reports/m5/traces_unified/episodes.jsonl  (real BIRD rollout lengths)
  --scenario  batch x n pairs to evaluate

The model is linear in generated/updated tokens per step, which is how the
observed M0 step decomposes (gen / old_log_prob / ref / update_actor).
"""
import argparse
import json
import statistics as st
from pathlib import Path


def load_episodes(path):
    path = Path(path)
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def field(name):
        return st.mean(r.get(name, 0) for r in rows)
    return {"episodes": len(rows), "avg_tool_calls": field("tool_calls"),
            "avg_response_tokens": field("response_tokens"),
            "avg_latency_ms": field("latency_ms"),
            "avg_observation_tokens": field("observation_tokens")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0", default="reports/m4/verl_e2e_step1.json")
    p.add_argument("--episodes", default="reports/m5/traces_unified/episodes.jsonl")
    p.add_argument("--out", default="reports/m6/training_cost_estimate.json")
    a = p.parse_args()

    m0 = json.loads(Path(a.m0).read_text(encoding="utf-8"))["metrics"]
    episodes = load_episodes(a.episodes) or {"avg_response_tokens": 300, "avg_tool_calls": 3,
                                             "avg_latency_ms": 16000, "avg_observation_tokens": 350}
    # tokens measured in the M0 step
    m0_tokens = m0["perf/total_num_tokens"]
    per_token = {
        "gen_ms": m0["timing_s/gen"] * 1000 / max(m0_tokens, 1),
        "old_log_prob_ms": m0["timing_s/old_log_prob"] * 1000 / max(m0_tokens, 1),
        "ref_ms": m0["timing_s/ref"] * 1000 / max(m0_tokens, 1),
        "update_actor_ms": m0["timing_s/update_actor"] * 1000 / max(m0_tokens, 1),
    }
    traj_tokens = episodes["avg_response_tokens"] + episodes["avg_observation_tokens"]

    scenarios = []
    for batch, n in ((4, 4), (8, 4), (8, 8), (16, 4), (16, 8)):
        traj = batch * n
        tokens = traj * traj_tokens
        gen_s = tokens * per_token["gen_ms"] / 1000 * 0.75  # agent loop overlaps tool time
        old_s = tokens * per_token["old_log_prob_ms"] / 1000
        ref_s = tokens * per_token["ref_ms"] / 1000
        upd_s = tokens * per_token["update_actor_ms"] / 1000
        step_s = gen_s + old_s + ref_s + upd_s
        scenarios.append({
            "batch_prompts": batch, "rollouts_per_prompt": n, "trajectories": traj,
            "tokens_per_step": int(tokens),
            "gen_s": round(gen_s, 1), "old_log_prob_s": round(old_s, 1),
            "ref_s": round(ref_s, 1), "update_actor_s": round(upd_s, 1),
            "step_s": round(step_s, 1), "step_min": round(step_s / 60, 1),
            "hours_100_steps": round(step_s * 100 / 3600, 1),
            "hours_200_steps": round(step_s * 200 / 3600, 1),
            "hours_400_steps": round(step_s * 400 / 3600, 1),
        })

    report = {
        "measured": {"m0_step_s": m0["timing_s/step"], "m0_tokens": m0_tokens,
                     "ms_per_token": {k: round(v, 4) for k, v in per_token.items()},
                     "episode_profile": episodes},
        "assumptions": [
            "linear scaling in tokens per optimizer step",
            "generation overlaps tool execution (0.75 factor)",
            "excludes evaluation, checkpointing and startup",
        ],
        "scenarios": scenarios,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:1600])


if __name__ == "__main__":
    main()
