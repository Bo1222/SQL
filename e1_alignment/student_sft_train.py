"""M6-08m student-only SFT trainer.

Key differences from the old agent SFT trainer:

* the token-level ``trajectory.loss_mask`` is consumed directly;
* the mask is never rebuilt from ``segment.kind == "action"``;
* gradient accumulation uses the causal supervised-token mean over the whole
  window (``e1_alignment.rebuild.training_math.backward_window``), so there is
  no ``loss / grad_accum`` followed by ``grad / window`` double scaling.

The trainer is deliberately strict: one unusable record rejects the dataset
unless the caller inspects it with ``--dry-run`` first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e1_alignment.rebuild.training_math import backward_window, lr_scale  # noqa: E402
from e1_alignment.student_sft_data import (  # noqa: E402
    StudentDataset,
    collate,
    load_examples,
    masked_loss,
    summary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="0 disables the hard timeout; set a budget explicitly")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--require-trainable", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="default true; use --no-require-trainable only for debugging")
    parser.add_argument("--allow-audit-only", action="store_true",
                        help="explicit debug escape hatch for non-trainable provenance")
    parser.add_argument("--check-mask", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only data/mask validation; do not load a model")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.epochs >= 1 and args.grad_accum >= 1 and args.lr > 0):
        raise SystemExit("invalid training configuration")
    if Path(args.out).exists():
        raise SystemExit("candidate output already exists; no overwrite")

    require_trainable = args.require_trainable and not args.allow_audit_only
    rows, excluded = load_examples(args.data, max_context=args.max_context,
                                   require_trainable=require_trainable)
    data_summary = summary(rows, excluded)
    data_summary.update({
        "data": args.data,
        "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "require_trainable": require_trainable,
        "max_context": args.max_context,
    })
    print(json.dumps(data_summary, ensure_ascii=False), flush=True)
    if not rows or excluded:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "data_reject.json").write_text(
            json.dumps({**data_summary, "excluded_examples": excluded[:200]},
                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        raise SystemExit("dataset has unusable examples; reject entire dataset")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "loaded": len(rows),
                          "supervised_tokens": data_summary["supervised_tokens"]}),
              flush=True)
        return

    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    from torch.utils.data import DataLoader

    from e1_alignment.sft_common import build_model, mask_self_check

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_summary.json").write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    dataset = StudentDataset(rows)
    model, tokenizer = build_model(args.model, args.lora_rank, args.lora_alpha,
                                   args.lora_dropout, args.device)
    check = mask_self_check(model, dataset, args.device)
    (out / "mask_check.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(check, ensure_ascii=False), flush=True)
    if args.check_mask:
        return

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.smoke_steps or args.max_steps or steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: lr_scale(index, total_steps))

    class TrainingStopped(RuntimeError):
        pass

    class MaxStepsReached(RuntimeError):
        pass

    started = time.monotonic()

    def expired(signum, frame):
        raise TrainingStopped("training_hard_timeout")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)

    step = 0
    stop_reason = None
    log_path = out / "train_log.jsonl"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(args.epochs):
                iterator = iter(loader)
                while True:
                    if args.max_steps and step >= args.max_steps:
                        raise MaxStepsReached()
                    batches = list(islice(iterator, args.grad_accum))
                    if not batches:
                        break
                    loss, token_count = backward_window(model, batches, masked_loss)
                    norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0,
                                                                error_if_nonfinite=True))
                    used_lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = {
                        "step": step, "epoch": epoch + 1, "loss": loss,
                        "grad_norm": norm, "supervised_tokens": token_count,
                        "micro_batches": len(batches), "lr": used_lr,
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                    if args.smoke_steps and step >= args.smoke_steps:
                        raise TrainingStopped("smoke_steps")
                    if args.save_every and step % args.save_every == 0:
                        save_dir = out / f"step_{step}"
                        model.save_pretrained(save_dir, selected_adapters=["default"])
                        tokenizer.save_pretrained(save_dir)
    except MaxStepsReached:
        stop_reason = "max_steps"
    except (TrainingStopped, FloatingPointError, RuntimeError, ValueError) as exc:
        stop_reason = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    save_path = out / "interrupted" if stop_reason not in (None, "max_steps") else out
    model.save_pretrained(save_path, selected_adapters=["default"])
    tokenizer.save_pretrained(save_path)
    result = {
        "completed": stop_reason in (None, "max_steps"),
        "stop_reason": stop_reason,
        "model": str(Path(args.model).resolve()),
        "out": str(out),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps": step,
        "total_steps": total_steps,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_context": args.max_context,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout},
        "seed": args.seed,
        "data": data_summary,
        "normalization": "causal supervised-token mean over actual accumulation window",
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    (out / "train_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
