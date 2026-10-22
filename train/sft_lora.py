"""LoRA SFT for Spider Text-to-SQL on Ascend NPU."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SFTSet(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        self.skipped = 0
        for row in rows:
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(row["gold_sql"].strip(), add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(completion_ids) > max_len:
                self.skipped += 1
                continue
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
            attention[i, : len(ids)] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/spider/sft.jsonl")
    parser.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", default="checkpoints/sft_lora")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    started = time.time()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTSet(rows, tokenizer, args.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
        num_workers=0,
        drop_last=False,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(
        f"[data] rows={len(rows)} kept={len(dataset)} skipped_over_len={dataset.skipped} "
        f"micro_batches={len(loader)} steps={total_steps}",
        flush=True,
    )

    print("[model] loading", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.to("npu:0")
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[train] start {time.strftime('%Y-%m-%d %H:%M:%S')} steps={total_steps}", flush=True)
    global_step = 0
    micro = 0
    tokens_since_log = 0
    last_log = time.time()
    loss_sum = 0.0
    loss_count = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch in loader:
            batch = {key: value.to("npu:0") for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            loss_sum += float(outputs.loss.detach())
            loss_count += 1
            tokens_since_log += int(batch["attention_mask"].sum())
            micro += 1

            if micro % args.grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                now = time.time()
                elapsed = now - started
                steps_per_s = args.log_every / max(now - last_log, 1e-6) if global_step > 1 else 0
                print(
                    f"[step] {global_step}/{total_steps} loss={loss_sum / max(loss_count, 1):.4f} "
                    f"grad_norm={float(grad_norm):.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={elapsed:.0f}s steps_per_s={steps_per_s:.2f} "
                    f"tokens_per_s={tokens_since_log / max(now - last_log, 1e-6):.0f} "
                    f"mem_gb={torch.npu.max_memory_allocated() / 1e9:.1f}",
                    flush=True,
                )
                loss_sum, loss_count, tokens_since_log, last_log = 0.0, 0, 0, now

            if args.save_every and global_step % args.save_every == 0:
                ckpt = Path(args.out) / f"step_{global_step}"
                model.save_pretrained(ckpt)
                print(f"[save] {ckpt}", flush=True)

            if args.max_steps and global_step >= args.max_steps:
                print(f"[stop] reached max_steps={args.max_steps}", flush=True)
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    summary = {
        "data": args.data,
        "rows": len(rows),
        "kept": len(dataset),
        "skipped_over_len": dataset.skipped,
        "steps": global_step,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_len": args.max_len,
        "elapsed_s": round(time.time() - started, 1),
        "peak_mem_gb": round(torch.npu.max_memory_allocated() / 1e9, 2),
        "output": str(out),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
