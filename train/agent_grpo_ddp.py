"""Whole-trajectory observation-masked GRPO on CPU/gloo or Ascend/HCCL."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from agent.common import choose_device, load_metadata, load_model, read_jsonl, save_adapter, select_rows, write_json, write_jsonl
from agent.loop import AgentConfig, run_episode
from agent.reward import score_trajectory
from agent.rollout import HFGenerator
from agent.tools import DatabaseTools
from agent.trajectory import masked_logprob
from train.grpo_tracking import log_metrics, training_logs
from train.project_env import load_project_env


def group_advantages(rewards):
    values = torch.tensor(rewards, dtype=torch.float32)
    std = values.std(unbiased=False)
    return ((values - values.mean()) / (std + 1e-4)).tolist() if std > 0 else [0.] * len(rewards)


def sum_across(value, device, distributed):
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def parse_args():
    load_project_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "npu"], default="auto")
    p.add_argument("--steps", type=int, default=100, help="requested updates with nonzero group advantages")
    p.add_argument("--max-attempts", type=int, default=0, help="default: 3 * steps")
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tools", type=int, default=4)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--max-prompt", type=int, default=1536)
    p.add_argument("--max-generated", type=int, default=2048)
    p.add_argument("--max-action", type=int, default=512)
    p.add_argument("--max-observation", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=.001)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--val-data", default="")
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--static-retrieval", action="store_true", help="single-turn RL control: fixed metadata retrieval")
    p.add_argument("--pilot-report", default="", help="optional cold-start coverage gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default=os.environ.get("WANDB_MODE", "offline"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sqlrl-agent"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-dir", default=None)
    a = p.parse_args()
    if a.pilot_report:
        report = json.loads(Path(a.pilot_report).read_text())
        if (not report.get("ready_for_rl") or Path(report["adapter"]).resolve() != Path(a.adapter).resolve()
                or Path(report["data"]).resolve() != Path(a.data).resolve()):
            p.error("pilot is not ready or was produced by a different adapter; fix cold start before RL")
    if min(a.steps, a.prompts_per_step, a.val_every, a.patience) < 1 or a.n < 2 or min(a.max_attempts, a.save_every, a.val_limit) < 0:
        p.error("steps/prompts must be positive, n>=2, attempt/save budgets nonnegative")
    if a.lr <= 0 or a.kl_coef < 0 or not 0 <= a.clip < 1:
        p.error("invalid optimization parameters")
    AgentConfig(**{k: getattr(a, k) for k in AgentConfig.__dataclass_fields__})
    if (Path(a.out) / "adapter_config.json").exists():
        p.error("output already contains an adapter; choose a new output directory")
    return a


def train(a, rank, world, device, distributed, log_file, run):
    rows = read_jsonl(a.data)
    if not rows or any(r.get("split") != "rl" for r in rows):
        raise ValueError("RL requires a nonempty split=rl dataset; SFT/val/test are forbidden")
    validation = select_rows(read_jsonl(a.val_data), a.val_limit, 0) if a.val_data else []
    if a.val_data and (not validation or any(r.get("split") != "val" for r in validation)):
        raise ValueError("checkpoint selection only accepts a nonempty split=val dataset")
    if {r["db_id"] for r in validation} & {r["db_id"] for r in rows}:
        raise ValueError("validation databases must be disjoint from training databases")
    torch.manual_seed(a.seed + rank)
    rng = random.Random(a.seed)  # same prompt sample on every rank, independent of tracker RNG
    model, tok = load_model(a.model, a.adapter, device, trainable=True, reference=True)
    model.set_adapter("policy")
    params = [p for name, p in model.named_parameters() if "policy" in name and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.)
    config = AgentConfig(**{k: getattr(a, k) for k in AgentConfig.__dataclass_fields__})
    generator = HFGenerator(model, tok, 1.)
    started = time.monotonic()
    updates = optimizer_steps = attempts = 0
    best_ex, stale = -1., 0
    early_stopped = False
    while updates < a.steps and attempts < (a.max_attempts or 3 * a.steps):
        attempts += 1
        step_started = time.monotonic()
        batch = rng.sample(rows, min(a.prompts_per_step, len(rows)))
        trajectories = []
        zero = full = mixed = infra = 0
        for row in batch[rank::world]:
            model.set_adapter("policy")
            episodes, rewards = [], []
            for _ in range(a.n):
                tools = DatabaseTools(row, load_metadata(row))
                context = ""
                if a.static_retrieval:
                    from eval.eval_agent import static_retrieval
                    context = static_retrieval(row, tok, tools, config)
                trajectory = run_episode(row, tok, generator, tools, config, context)
                if a.static_retrieval:
                    trajectory.tool_calls += 1
                result = score_trajectory(row, trajectory)
                episodes.append((trajectory, result))
                rewards.append(result["reward"])
            # Drop the whole question group on infrastructure failure; never relabel it wrong.
            if any(r.get("infrastructure_error") for _, r in episodes):
                infra += 1
                continue
            advantages = group_advantages(rewards)
            zero += int(all(r == 0 for r in rewards))
            full += int(all(r == 1 for r in rewards))
            mixed += int(any(v != 0 for v in advantages))
            trajectories.extend((trajectory, reward, adv) for (trajectory, reward), adv in zip(episodes, advantages)
                                if any(trajectory.loss_mask))
        stats = sum_across([len(trajectories), sum(r["reward"] for _, r, _ in trajectories),
                           sum(r["ok"] for _, r, _ in trajectories), zero, full, mixed, infra,
                           sum(t.tool_calls for t, _, _ in trajectories),
                           sum(t.generated_tokens for t, _, _ in trajectories),
                           sum(t.invalid_actions for t, _, _ in trajectories)], device, distributed)
        count, reward_sum, valid, zero, full, mixed, infra, calls, tokens, invalid = stats
        loss_sum = kl_sum = pg_sum = grad = 0.
        optimizer.zero_grad(set_to_none=True)
        if count and (mixed or a.kl_coef):
            # Old/ref scores use exact stored histories and the same action mask.
            scored = []
            for trajectory, _, adv in trajectories:
                old = masked_logprob(model, trajectory.input_ids, trajectory.loss_mask, "policy")
                ref = masked_logprob(model, trajectory.input_ids, trajectory.loss_mask, "ref")
                scored.append((trajectory, adv, old, ref))
            model.set_adapter("policy")
            model.train()  # checkpointing must be active at forward AND recomputation
            for trajectory, adv, old, ref in scored:
                logp = masked_logprob(model, trajectory.input_ids, trajectory.loss_mask, "policy", True)
                ratio = (logp - old).exp()
                pg = -torch.minimum(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
                delta = ref - logp
                kl = (delta.exp() - delta - 1).mean()
                loss = pg + a.kl_coef * kl
                (loss / count).backward()
                loss_sum += float(loss.detach()); kl_sum += float(kl.detach()); pg_sum += float(pg.detach())
            for param in params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                if distributed:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            optimizer.step()
            optimizer_steps += 1
            updates += int(mixed > 0)
            model.eval()
        totals = sum_across([loss_sum, kl_sum, pg_sum], device, distributed)
        memory = torch.tensor(torch.npu.max_memory_allocated() / 1e9 if device.startswith("npu") else 0., device=device)
        if distributed:
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            record = {"step": attempts, "effective_updates": updates, "optimizer_steps": optimizer_steps,
                      "reward_mean": reward_sum / max(count, 1), "valid_rate": valid / max(count, 1),
                      "trajectories": count, "all_zero_groups": zero, "all_correct_groups": full,
                      "mixed_groups": mixed, "infrastructure_groups": infra,
                      "tool_calls_mean": calls / max(count, 1), "generated_tokens_mean": tokens / max(count, 1),
                      "invalid_actions": invalid, "loss": totals[0] / max(count, 1),
                      "kl": totals[1] / max(count, 1), "pg_loss": totals[2] / max(count, 1),
                      "grad_norm": grad, "lr": a.lr, "mem_gb": float(memory),
                      "kl_only": mixed == 0 and count > 0 and a.kl_coef > 0,
                      "step_s": time.monotonic() - step_started, "elapsed_s": time.monotonic() - started}
            log_metrics(log_file, run, record)
            print(json.dumps(record), flush=True)
        if mixed and a.save_every and updates % a.save_every == 0:
            if rank == 0:
                save_adapter(model, tok, Path(a.out) / f"update_{updates}")
            if distributed:
                dist.barrier()
        if mixed and validation and updates % a.val_every == 0:
            stop = torch.zeros(1, device=device)
            if rank == 0:
                greedy = HFGenerator(model, tok, 0.)
                evaluated = []
                for row in validation:
                    tools = DatabaseTools(row, load_metadata(row))
                    context = ""
                    if a.static_retrieval:
                        from eval.eval_agent import static_retrieval
                        context = static_retrieval(row, tok, tools, config)
                    trajectory = run_episode(row, tok, greedy, tools, config, context)
                    result = score_trajectory(row, trajectory)
                    if result.get("infrastructure_error"):
                        raise RuntimeError("Validation infrastructure failure; do not use it for checkpoint selection")
                    evaluated.append({"id": row["id"], "db_id": row["db_id"], "question": row["question"],
                                      "gold_sql": row["gold_sql"], "final": result, "trajectory": trajectory.to_dict()})
                ex = sum(r["final"]["correct"] for r in evaluated) / len(evaluated)
                write_jsonl(Path(a.out) / f"validation_{updates}.jsonl", evaluated)
                if ex >= best_ex + .005:
                    best_ex, stale = ex, 0
                    save_adapter(model, tok, Path(a.out) / "best")
                    write_json(Path(a.out) / "best.json", {"effective_updates": updates, "validation_ex": ex,
                                                            "validation_data": a.val_data, "validation_n": len(validation)})
                else:
                    stale += 1
                print(f"[validation] update={updates} ex={ex:.4f} best={best_ex:.4f} stale={stale}", flush=True)
                if run is not None:
                    run.log({"validation_ex": ex, "best_validation_ex": best_ex}, step=attempts)
                stop.fill_(stale >= a.patience)
            if distributed:
                dist.broadcast(stop, src=0)
            if stop.item():
                early_stopped = True
                break
    if rank == 0:
        save_adapter(model, tok, a.out)
        summary = {**vars(a), "world_size": world, "attempts": attempts, "effective_updates": updates,
                   "optimizer_steps": optimizer_steps, "requested_updates_reached": updates >= a.steps,
                   "early_stopped": early_stopped, "best_validation_ex": best_ex if validation else None,
                   "elapsed_s": time.monotonic() - started}
        write_json(Path(a.out) / "train_summary.json", summary)
        if run is not None:
            run.summary.update(summary)
    if distributed:
        dist.barrier()


def main():
    a = parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = choose_device(a.device, int(os.environ.get("LOCAL_RANK", 0)))
    distributed = world > 1
    if distributed:
        dist.init_process_group("hccl" if device.startswith("npu") else "gloo")
    try:
        with training_logs(a, rank, world) as (log_file, run):
            train(a, rank, world, device, distributed, log_file, run)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
