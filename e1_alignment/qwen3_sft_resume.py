"""M6-08l L2 approximate resume: continue Qwen3-4B LoRA SFT from step 91 to 105.

The partial run saved only LoRA adapter weights at step 91; optimizer, scheduler and
dropout RNG state were not saved. This script therefore:
  - loads the step-91 adapter (weights are continuous);
  - reconstructs the seed=0 DataLoader order and validates it against the partial log;
  - runs the remaining 14 steps (92..105) on the remaining 112 examples;
  - reconstructs the original LR schedule (lr_scale(step-1, 105));
  - uses a fresh AdamW and a reseeded RNG.

It is an approximate continuation, not a bit-exact resume. The summary records this.
"""
import argparse
import hashlib
import json
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from e1_alignment.common import write_json
from e1_alignment.rebuild.training_math import backward_window, lr_scale
from e1_alignment.sft_train import (
    ExampleDataset,
    collate,
    load_examples,
    masked_loss,
)


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def seed0_order(n):
    """Reproduce the original DataLoader(shuffle=True, generator seed=0) order."""
    class _Indices(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, index):
            return index

    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(_Indices(), batch_size=1, shuffle=True, generator=generator)
    return [int(index) for index in loader]


def build_resume_model(model_path, adapter_path, device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    # PEFT may keep the saved adapter frozen; force LoRA params trainable.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("no_trainable_lora_params")
    b_weights = [param for name, param in model.named_parameters() if "lora_B" in name]
    if not b_weights or not any(bool(torch.count_nonzero(param)) for param in b_weights):
        raise RuntimeError("adapter_B_all_zero__not_a_trained_adapter")
    return model, tokenizer, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/m6/qwen3_4b_sft/e1_sft_qwen3.jsonl")
    parser.add_argument("--model",
                        default="/home/ma-user/work/env_verify/models/Qwen3-4B-Instruct-2507-ms")
    parser.add_argument("--adapter", default="outputs/qwen3_4b_sft/e1_lora_v1_partial91")
    parser.add_argument("--partial-log", default="logs/qwen3_4b_sft_l2_partial91.log")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-step", type=int, default=91)
    parser.add_argument("--total-steps", type=int, default=105)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    import numpy as np
    import torch_npu  # noqa: F401

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows, excluded = load_examples(args.data, args.max_context)
    if len(rows) != 840 or excluded:
        raise RuntimeError(f"unexpected_data_filter kept={len(rows)} excluded={len(excluded)}")
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()

    order = seed0_order(len(rows))
    consumed = args.start_step * args.grad_accum
    remaining_rows = [rows[index] for index in order[consumed:]]
    expected_remaining = (args.total_steps - args.start_step) * args.grad_accum
    if len(remaining_rows) != expected_remaining:
        raise RuntimeError(
            f"remaining_count_mismatch got={len(remaining_rows)} expected={expected_remaining}")

    # Validate the reconstructed order against the partial training log at several steps.
    partial = {record["step"]: record for record in read_jsonl(args.partial_log)
               if "step" in record}
    checks = {}
    for step in (1, 2, 3, 50, 90, args.start_step):
        chunk = order[(step - 1) * args.grad_accum:step * args.grad_accum]
        tokens = sum(int(sum(rows[index]["loss_mask"][1:])) for index in chunk)
        logged = int(partial[step]["supervised_tokens"])
        checks[str(step)] = {"reconstructed": tokens, "logged": logged,
                             "match": tokens == logged}
        if tokens != logged:
            raise RuntimeError(f"data_order_check_failed step={step} {tokens} != {logged}")

    print(json.dumps({"data": args.data, "sha256": digest, "kept": len(rows),
                      "excluded": len(excluded), "start_step": args.start_step,
                      "remaining_examples": len(remaining_rows),
                      "order_checks": checks}, ensure_ascii=False), flush=True)

    model, tokenizer, params = build_resume_model(args.model, args.adapter, args.device)
    print(json.dumps({"adapter": args.adapter, "trainable_params": len(params),
                      "trainable_elems": sum(p.numel() for p in params),
                      "model_training": model.training}, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    dataset = ExampleDataset(remaining_rows)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    iterator = iter(loader)

    started = time.monotonic()
    stop_reason = None
    hbm_peak = 0.0
    losses, grad_norms, lrs, tokens_per_step = [], [], [], []
    log_path = out / "train_log.jsonl"
    step = args.start_step
    try:
        with log_path.open("w", encoding="utf-8") as log:
            optimizer.zero_grad(set_to_none=True)
            for step in range(args.start_step + 1, args.total_steps + 1):
                if time.monotonic() - started >= args.max_seconds:
                    raise RuntimeError("resume_hard_timeout")
                batches = list(islice(iterator, args.grad_accum))
                if len(batches) != args.grad_accum:
                    raise RuntimeError(f"short_window step={step} got={len(batches)}")
                used_lr = args.lr * lr_scale(step - 1, args.total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = used_lr
                loss, token_count = backward_window(model, batches, masked_loss)
                norm = float(torch.nn.utils.clip_grad_norm_(
                    params, 1.0, error_if_nonfinite=True))
                if norm <= 0.0:
                    raise RuntimeError(f"zero_grad_norm step={step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                try:
                    hbm = torch_npu.npu.max_memory_allocated() / 1024 ** 3
                    hbm_peak = max(hbm_peak, hbm)
                except Exception:
                    hbm = None
                record = {"step": step, "global_step": step, "resume_step": step - args.start_step,
                          "loss": loss, "grad_norm": norm, "supervised_tokens": token_count,
                          "micro_batches": len(batches), "lr": used_lr, "hbm_gb": hbm,
                          "elapsed_s": round(time.monotonic() - started, 1)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                losses.append(loss)
                grad_norms.append(norm)
                lrs.append(used_lr)
                tokens_per_step.append(token_count)
    except RuntimeError as exc:
        stop_reason = str(exc)
    finally:
        elapsed = round(time.monotonic() - started, 1)
        model.save_pretrained(out, selected_adapters=["default"])
        tokenizer.save_pretrained(out)
        summary = {
            "stage": "M6-08l-L2-resume",
            "completed": stop_reason is None,
            "stop_reason": stop_reason,
            "resume_from": args.adapter,
            "resume_start_step": args.start_step,
            "resume_end_step": args.total_steps,
            "steps_run": step - args.start_step,
            "last_step": step,
            "optimizer_state_recovered": False,
            "scheduler_state_recovered": False,
            "rng_state_recovered": False,
            "data_order_recovered": True,
            "data_order_checks": checks,
            "training_math": "token_weighted_accum",
            "fresh_optimizer": "AdamW(lr=2e-6, weight_decay=0.0)",
            "lr_schedule": "lr_scale(global_step-1, 105), warmup ceil(5%)=6",
            "seed": args.seed,
            "data": args.data,
            "data_sha256": digest,
            "model": args.model,
            "trainable_params": len(params),
            "trainable_elems": sum(p.numel() for p in params),
            "steps": args.total_steps,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_context": args.max_context,
            "elapsed_s": elapsed,
            "hbm_peak_gb": hbm_peak or None,
            "losses": losses,
            "grad_norms": grad_norms,
            "lrs": lrs,
            "supervised_tokens": tokens_per_step,
            "train_log": str(log_path),
            "note": ("Approximate continuation: adapter weights are continuous from step 91, "
                     "but optimizer moments and dropout RNG were not saved; data order and LR "
                     "schedule are reconstructed. Not a bit-exact resume."),
        }
        write_json(out / "train_summary.json", summary)
        print(json.dumps({"done": stop_reason is None, "steps_run": step - args.start_step,
                          "last_step": step, "elapsed_s": elapsed,
                          "hbm_peak_gb": hbm_peak or None}), flush=True)
        if stop_reason:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
