"""M6-08l L1/L2: Qwen3-4B LoRA SFT on Qwen3-tokenized E1 text.

This is a controlled model/data comparison against the 3B SFT baseline.
It uses token-weighted accumulation (the fixed helper), not the old
double-normalized E1 trainer.
"""
import argparse
import hashlib
import json
import math
import os
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from e1_alignment.common import write_json
from e1_alignment.rebuild.training_math import backward_window, lr_scale
from e1_alignment.sft_train import (
    ExampleDataset,
    build_model,
    collate,
    load_examples,
    mask_self_check,
    masked_loss,
)


def main():
    p = argparse.ArgumentParser()
    model_root = Path(os.environ.get("SQLRL_MODEL_ROOT", Path.home() / "work" / "env_verify" / "models"))
    p.add_argument("--data", default="artifacts/m6/qwen3_4b_sft/e1_sft_qwen3.jsonl")
    p.add_argument("--model", default=str(model_root / "Qwen3-4B-Instruct-2507-ms"))
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--check-mask", action="store_true")
    p.add_argument("--no-final-save", action="store_true")
    a = p.parse_args()

    import numpy as np
    import torch_npu  # noqa: F401

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    rows, excluded = load_examples(a.data, a.max_context)
    if not rows:
        raise SystemExit("no usable examples")
    dataset = ExampleDataset(rows)
    digest = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    print(json.dumps({"data": a.data, "sha256": digest, "kept": len(rows),
                      "excluded": len(excluded)}), flush=True)
    model, tokenizer = build_model(a.model, a.lora_rank, a.lora_alpha, a.lora_dropout, a.device)
    check = mask_self_check(model, dataset, a.device)
    print(json.dumps(check), flush=True)
    if a.check_mask:
        write_json(Path(a.out) / "mask_check.json", check)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(a.seed))
    steps_per_epoch = math.ceil(len(loader) / a.grad_accum)
    total_steps = a.smoke_steps or a.max_steps or steps_per_epoch * a.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: lr_scale(index, total_steps))
    started = time.monotonic()
    step = 0
    stop_reason = None
    hbm_peak = 0.0
    log_path = out / "train_log.jsonl"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(a.epochs):
                iterator = iter(loader)
                while True:
                    if time.monotonic() - started >= a.max_seconds:
                        raise RuntimeError("training_hard_timeout")
                    batches = list(islice(iterator, a.grad_accum))
                    if not batches:
                        break
                    loss, token_count = backward_window(model, batches, masked_loss)
                    norm = float(torch.nn.utils.clip_grad_norm_(
                        params, 1.0, error_if_nonfinite=True))
                    used_lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    try:
                        hbm = torch_npu.npu.max_memory_allocated() / 1024 ** 3
                        hbm_peak = max(hbm_peak, hbm)
                    except Exception:
                        hbm = None
                    record = {"step": step, "epoch": epoch + 1, "loss": loss,
                              "grad_norm": norm, "supervised_tokens": token_count,
                              "micro_batches": len(batches), "lr": used_lr,
                              "hbm_gb": hbm,
                              "elapsed_s": round(time.monotonic() - started, 1)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                    if a.save_every and step % a.save_every == 0:
                        checkpoint = out / f"step_{step}"
                        model.save_pretrained(checkpoint, selected_adapters=["default"])
                        tokenizer.save_pretrained(checkpoint)
                    if step >= total_steps:
                        break
                if step >= total_steps:
                    break
    except RuntimeError as exc:
        stop_reason = str(exc)
    finally:
        elapsed = round(time.monotonic() - started, 1)
        if not a.no_final_save:
            model.save_pretrained(out, selected_adapters=["default"])
            tokenizer.save_pretrained(out)
        summary = {
            "stage": "M6-08l-L1" if (a.smoke_steps or a.max_steps) else "M6-08l-L2",
            "completed": stop_reason is None, "stop_reason": stop_reason,
            "data": a.data, "data_sha256": digest,
            "kept": len(rows), "excluded": len(excluded),
            "model": a.model, "initialization": "Qwen3-4B-Instruct-2507 + fresh zero-init LoRA",
            "token_weighted_accum": True,
            "epochs": a.epochs, "steps": step, "grad_accum": a.grad_accum, "lr": a.lr,
            "max_context": a.max_context,
            "lora": {"r": a.lora_rank, "alpha": a.lora_alpha, "dropout": a.lora_dropout},
            "elapsed_s": elapsed, "hbm_peak_gb": hbm_peak or None,
            "mask_check": check, "train_log": str(log_path),
        }
        write_json(out / "train_summary.json", summary)
        write_json(out / "excluded.json", excluded)
        print(json.dumps({"done": stop_reason is None, "steps": step,
                          "elapsed_s": elapsed, "hbm_peak_gb": hbm_peak or None}))
        if stop_reason:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
