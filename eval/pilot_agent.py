"""Training-only exploration pilot: diagnose pass@N and group reward coverage."""
import argparse
import json
from collections import Counter
from pathlib import Path

from agent.common import choose_device, load_metadata, load_model, read_jsonl, select_rows, write_json
from agent.loop import AgentConfig, run_episode
from agent.reward import score_trajectory
from agent.rollout import HFGenerator
from agent.tools import DatabaseTools


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--adapter", required=True)
    p.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=200); p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "npu"], default="auto")
    for key, field in AgentConfig.__dataclass_fields__.items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=field.default)
    a = p.parse_args()
    if a.n < 2 or a.limit < 0:
        p.error("pilot requires n>=2 and a nonnegative limit")
    rows = read_jsonl(a.data)
    if not rows or any(r.get("split") != "rl" for r in rows):
        p.error("difficulty/coverage pilot only accepts the RL training split")
    rows = select_rows(rows, a.limit, a.seed)
    import torch
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, a.adapter, choose_device(a.device))
    generator = HFGenerator(model, tok, 1.)
    config = AgentConfig(**{k: getattr(a, k) for k in AgentConfig.__dataclass_fields__})
    totals = Counter()
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            episodes, rewards = [], []
            for _ in range(a.n):
                trajectory = run_episode(row, tok, generator, DatabaseTools(row, load_metadata(row)), config)
                result = score_trajectory(row, trajectory)
                episodes.append({"trajectory": trajectory.to_dict(), "final": result})
                rewards.append(result["reward"])
                totals["episodes"] += 1
                totals["invalid_actions"] += trajectory.invalid_actions
                totals["actions"] += sum(s.kind == "action" for s in trajectory.segments)
                totals["submitted"] += int(trajectory.final_sql is not None)
            infra = any(e["final"].get("infrastructure_error") for e in episodes)
            totals["infrastructure_groups"] += int(infra)
            if not infra:
                totals["groups"] += 1
                totals["mixed_groups"] += int(0 < sum(rewards) < a.n)
                totals["all_zero_groups"] += int(sum(rewards) == 0)
                totals["all_correct_groups"] += int(sum(rewards) == a.n)
                totals["pass_groups"] += int(sum(rewards) > 0)
                totals["correct_episodes"] += sum(rewards)
            stream.write(json.dumps({"id": row["id"], "db_id": row["db_id"], "episodes": episodes}) + "\n")
            stream.flush()
            print(f"[pilot] groups={totals['groups']} mixed={totals['mixed_groups']} pass={totals['pass_groups']}", flush=True)
    summary = {**vars(a), **dict(totals), "pass_at_n": totals["pass_groups"] / max(totals["groups"], 1),
               "mixed_group_ratio": totals["mixed_groups"] / max(totals["groups"], 1),
               "legal_action_rate": 1 - totals["invalid_actions"] / max(totals["actions"], 1),
               "submission_rate": totals["submitted"] / max(totals["episodes"], 1)}
    summary["ready_for_rl"] = bool(summary["legal_action_rate"] >= .95 and totals["mixed_groups"] > 0
                                   and totals["infrastructure_groups"] == 0)
    write_json(a.out + ".summary.json", summary)


if __name__ == "__main__":
    main()
