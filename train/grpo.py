"""GRPO for Spider Text-to-SQL, initialized from an SFT LoRA adapter.

Reference policy is the frozen SFT adapter (second PEFT adapter, not the base model).
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402
from reward.exec_reward import audit_flags, compare, extract_sql  # noqa: E402
from train.grpo_utils import generate_group, sequence_logprob  # noqa: E402


def save_policy(model, path):
    """Save the policy adapter and flatten PEFT's per-adapter subdirectory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.set_adapter("policy")
    model.save_pretrained(path, selected_adapters=["policy"])
    sub = path / "policy"
    if sub.is_dir():
        for item in sub.iterdir():
            item.rename(path / item.name)
        sub.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/spider/rl.jsonl")
    parser.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft-adapter", default="checkpoints/sft_lora")
    parser.add_argument("--out", default="checkpoints/grpo")
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.set_device(0)

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[data] rl_prompts={len(rows)} steps={args.steps} n={args.n}", flush=True)
    print("[model] loading base + SFT adapter", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.sft_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    model.to("npu:0")
    model.eval()
    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = "policy" in name
    trainable = [p for n, p in model.named_parameters() if "lora_" in n and "policy" in n and p.requires_grad]
    print(f"[model] trainable policy params={sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_file = log_path.open("w", encoding="utf-8")
    started = time.time()
    print(f"[train] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for step in range(1, args.steps + 1):
        step_start = time.time()
        prompts = random.sample(rows, min(args.prompts_per_step, len(rows)))

        kept, zero_var, rewards, valid = [], 0, [], 0
        kept_group_count = 0
        for row in prompts:
            samples = generate_group(
                model, tokenizer, row["prompt"], args.n, args.max_new_tokens,
                args.temperature, args.max_prompt_tokens,
            )
            gold = execute(row["db_path"], row["gold_sql"])
            group_rewards = []
            for sample in samples:
                if int(sample["ids"].numel() - sample["start"]) < 2:
                    sample.update({"reward": 0.0, "adv": 0.0, "valid": False, "correct": False})
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
            kept_group_count += 1
            mean = float(values.mean())
            for sample, reward in zip(samples, group_rewards):
                sample["adv"] = (reward - mean) / (std + 1e-4)
                if sample["adv"] != 0.0:
                    kept.append(sample)

        if not kept:
            print(f"[step] {step}/{args.steps} all_zero_var skip elapsed={time.time() - step_start:.0f}s", flush=True)
            continue

        with torch.no_grad():
            for sample in kept:
                sample["old_logp"] = sequence_logprob(model, sample["ids"], sample["start"], "policy", False)
                sample["ref_logp"] = sequence_logprob(model, sample["ids"], sample["start"], "ref", False)

        optimizer.zero_grad(set_to_none=True)
        loss_sum = kl_sum = 0.0
        for sample in kept:
            logp = sequence_logprob(model, sample["ids"], sample["start"], "policy", True)
            adv = torch.tensor(sample["adv"], dtype=torch.float32, device=logp.device)
            ratio = torch.exp(logp - sample["old_logp"])
            pg_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
            log_ratio = sample["ref_logp"] - logp
            kl = torch.exp(log_ratio) - log_ratio - 1.0
            loss = pg_loss + args.kl_coef * kl.mean()
            (loss / len(kept)).backward()
            loss_sum += float(loss.detach())
            kl_sum += float(kl.mean().detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            record = {
                "step": step,
                "reward_mean": round(sum(rewards) / max(len(rewards), 1), 4),
                "valid_rate": round(valid / max(len(rewards), 1), 4),
                "kept_groups": kept_group_count,
                "kept_samples": len(kept),
                "zero_var_groups": zero_var,
                "loss": round(loss_sum / max(len(kept), 1), 4),
                "kl": round(kl_sum / max(len(kept), 1), 4),
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                "mem_gb": round(torch.npu.max_memory_allocated() / 1e9, 1),
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            print(
                f"[step] {step}/{args.steps} reward={record['reward_mean']:.3f} "
                f"valid={record['valid_rate']:.3f} kept={record['kept_samples']} "
                f"zero_var={zero_var} loss={record['loss']:.4f} kl={record['kl']:.4f} "
                f"grad={record['grad_norm']:.3f} elapsed={record['elapsed_s']:.0f}s "
                f"mem={record['mem_gb']:.1f}",
                flush=True,
            )

        if args.save_every and step % args.save_every == 0:
            ckpt = out / f"step_{step}"
            save_policy(model, ckpt)
            print(f"[save] {ckpt}", flush=True)

    log_file.close()
    model.set_adapter("policy")
    model.delete_adapter("ref")
    save_policy(model, out)
    summary = {
        "data": args.data,
        "sft_adapter": args.sft_adapter,
        "steps": args.steps,
        "prompts_per_step": args.prompts_per_step,
        "n": args.n,
        "lr": args.lr,
        "kl_coef": args.kl_coef,
        "elapsed_s": round(time.time() - started, 1),
        "peak_mem_gb": round(torch.npu.max_memory_allocated() / 1e9, 2),
        "output": str(out),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
