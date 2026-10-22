"""Multi-NPU GRPO for Spider Text-to-SQL (HCCL + torchrun).

Same hyper-parameters as train/grpo.py. Each rank handles a disjoint slice of the
per-step prompts and gradients are all-reduced across ranks.

Launch:
    torchrun --standalone --nproc_per_node=4 -- train/grpo_ddp.py ...
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402
from reward.exec_reward import audit_flags, compare, extract_sql  # noqa: E402
from train.grpo_utils import generate_group, sequence_logprob  # noqa: E402
from train.grpo_tracking import log_metrics, training_logs  # noqa: E402
from train.project_env import load_project_env  # noqa: E402


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://")
    return local_rank, rank, world_size


def save_policy(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.set_adapter("policy")
    model.save_pretrained(path, selected_adapters=["policy"])
    sub = path / "policy"
    if sub.is_dir():
        for item in sub.iterdir():
            item.rename(path / item.name)
        sub.rmdir()


def parse_args():
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--data", default="data/processed/spider/rl.jsonl")
    parser.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft-adapter", default="checkpoints/sft_lora")
    parser.add_argument("--out", default="checkpoints/grpo_ddp")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"],
                        default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sqlrl"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME") or None)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR") or None)
    args = parser.parse_args()
    for name in ["steps", "prompts_per_step", "max_new_tokens", "max_prompt_tokens", "log_every"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n < 2:
        parser.error("--n must be at least 2 for group-relative advantages")
    if args.temperature <= 0 or args.lr <= 0 or args.kl_coef < 0:
        parser.error("temperature/lr must be positive and kl-coef must be nonnegative")
    if not 0 <= args.clip < 1 or args.save_every < 0:
        parser.error("clip must be in [0, 1) and save-every must be nonnegative")
    return args


def run_training(args, local_rank, rank, world_size, log_file, wandb_run):
    is_main = rank == 0
    device = torch.device(f"npu:{local_rank}")
    prompt_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training data is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"[dist] world_size={world_size} local_rank={local_rank}", flush=True)
        print(f"[data] rl_prompts={len(rows)} steps={args.steps} n={args.n}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.sft_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    model.to(device)
    model.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = "policy" in name
    trainable = [p for n, p in model.named_parameters() if "lora_" in n and "policy" in n and p.requires_grad]
    if is_main:
        print(f"[model] trainable params={sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out = Path(args.out)
    started = time.time()
    if is_main:
        print(f"[train] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    optimizer_steps = skipped_steps = 0
    for step in range(1, args.steps + 1):
        step_start = time.time()
        used_lr = optimizer.param_groups[0]["lr"]
        model.set_adapter("policy")
        model.eval()
        # This private RNG is unaffected by rank-zero tracking or other libraries.
        prompts = prompt_rng.sample(rows, min(args.prompts_per_step, len(rows)))
        local_prompts = prompts[rank::world_size]

        kept, rewards, valid, zero_var = [], [], 0, 0
        for row in local_prompts:
            samples = generate_group(
                model, tokenizer, row["prompt"], args.n, args.max_new_tokens,
                args.temperature, args.max_prompt_tokens, device,
            )
            torch.npu.empty_cache()
            gold = execute(row["db_path"], row["gold_sql"])
            group_rewards = []
            for sample in samples:
                if int(sample["ids"].numel() - sample["start"]) < 2:
                    sample.update({"reward": 0.0, "valid": False, "correct": False})
                    group_rewards.append(0.0)
                    rewards.append(0.0)
                    continue
                sql, _ = extract_sql(sample["text"])
                sample["valid"] = False
                sample["correct"] = False
                hard = []
                if sql is not None:
                    pred = execute(row["db_path"], sql)
                    sample["valid"] = pred.ok
                    if pred.ok and gold.ok:
                        sample["correct"], _ = compare(gold.rows, pred.rows, row["gold_sql"])
                    if sample["valid"]:
                        valid += 1
                    hard, _ = audit_flags(
                        sql, row["gold_sql"], gold.rows if gold.ok else [], pred.rows if pred.ok else []
                    )
                sample["reward"] = 1.0 if sample["correct"] and not hard else 0.0
                group_rewards.append(sample["reward"])
                rewards.append(sample["reward"])

            values = torch.tensor(group_rewards, dtype=torch.float32)
            std = float(values.std(unbiased=False))
            if std == 0.0:
                zero_var += 1
                continue
            mean = float(values.mean())
            for sample, reward in zip(samples, group_rewards):
                sample["adv"] = (reward - mean) / (std + 1e-4)
                if sample["adv"] != 0.0:
                    kept.append(sample)

        stats = torch.tensor(
            [len(kept), sum(rewards), float(len(rewards)), float(valid), float(zero_var)],
            dtype=torch.float32, device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_kept = int(stats[0].item())
        loss_sum = kl_sum = 0.0
        grad_norm = 0.0
        if global_kept == 0:
            skipped_steps += 1
        else:
            model.eval()
            with torch.no_grad():
                for sample in kept:
                    sample["old_logp"] = sequence_logprob(model, sample["ids"], sample["start"], "policy", False)
                    sample["ref_logp"] = sequence_logprob(model, sample["ids"], sample["start"], "ref", False)

            optimizer.zero_grad(set_to_none=True)
            # Gradient checkpointing is only active in training mode.
            model.train()
            for sample in kept:
                logp = sequence_logprob(model, sample["ids"], sample["start"], "policy", True)
                adv = torch.tensor(sample["adv"], dtype=torch.float32, device=logp.device)
                ratio = torch.exp(logp - sample["old_logp"])
                pg_loss = -torch.min(
                    ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                ).mean()
                log_ratio = sample["ref_logp"] - logp
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                loss = pg_loss + args.kl_coef * kl.mean()
                (loss / global_kept).backward()
                loss_sum += float(loss.detach())
                kl_sum += float(kl.mean().detach())

            for param in trainable:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            model.eval()
            torch.npu.empty_cache()

        if step % args.log_every == 0 or step == args.steps or global_kept == 0:
            # Every rank participates, including ranks without retained samples.
            totals = torch.tensor([loss_sum, kl_sum], dtype=torch.float32, device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            peak_memory = torch.tensor(torch.npu.max_memory_allocated() / 1e9, device=device)
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            if is_main:
                record = {
                    "step": step,
                    "optimizer_steps": optimizer_steps,
                    "skipped": global_kept == 0,
                    "reward_mean": float(stats[1].item() / max(stats[2].item(), 1)),
                    "valid_rate": float(stats[3].item() / max(stats[2].item(), 1)),
                    "kept_samples": global_kept,
                    "zero_var_groups": int(stats[4].item()),
                    "loss": float(totals[0].item() / max(global_kept, 1)),
                    "kl": float(totals[1].item() / max(global_kept, 1)),
                    "grad_norm": grad_norm,
                    "lr": used_lr,
                    "next_lr": scheduler.get_last_lr()[0],
                    "step_s": time.time() - step_start,
                    "elapsed_s": time.time() - started,
                    "mem_gb": float(peak_memory.item()),
                }
                log_metrics(log_file, wandb_run, record)
                print(
                    f"[step] {step}/{args.steps} reward={record['reward_mean']:.3f} "
                    f"valid={record['valid_rate']:.3f} kept={global_kept} "
                    f"zero_var={record['zero_var_groups']} loss={record['loss']:.4f} "
                    f"kl={record['kl']:.4f} grad={record['grad_norm']:.3f} "
                    f"skipped={record['skipped']} elapsed={record['elapsed_s']:.0f}s "
                    f"mem={record['mem_gb']:.1f}",
                    flush=True,
                )

        if global_kept and args.save_every and step % args.save_every == 0:
            if is_main:
                save_policy(model, out / f"step_{step}")
                print(f"[save] {out / f'step_{step}'}", flush=True)
            dist.barrier()

    if is_main:
        model.set_adapter("policy")
        model.delete_adapter("ref")
        save_policy(model, out)
        summary = {
            "data": args.data,
            "sft_adapter": args.sft_adapter,
            "world_size": world_size,
            "steps": args.steps,
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "prompts_per_step": args.prompts_per_step,
            "n": args.n,
            "lr": args.lr,
            "kl_coef": args.kl_coef,
            "elapsed_s": round(time.time() - started, 1),
            "peak_mem_gb": round(record["mem_gb"], 2),
            "output": str(out),
        }
        (out / "train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if wandb_run is not None:
            wandb_run.summary.update(summary)
        print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    dist.barrier()


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = init_distributed()
    try:
        with training_logs(args, rank, world_size) as (log_file, wandb_run):
            run_training(args, local_rank, rank, world_size, log_file, wandb_run)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
