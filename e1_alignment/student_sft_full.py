"""M6-08m 3B full-parameter SFT with FSDP on 4 NPUs.

This is the full fine-tuning counterpart of ``student_sft_train.py`` (LoRA).
It keeps the same data contract and mask semantics:

* ``StudentDataset`` consumes the builder's ``trajectory.loss_mask``;
* ``backward_window`` uses the causal supervised-token mean;
* prompt/observation/failed-action tokens are never supervised.

Launch with torchrun, one process per NPU::

    torchrun --nproc_per_node=4 --master_port=29501 \
        -m e1_alignment.student_sft_full \
        --data artifacts/m6/e1_student_only/sft_student_full_prefer.jsonl \
        --out outputs/m6_08m/student_sft_full_3ep \
        --epochs 3 --grad-accum 2 --lr 2e-5 --smoke-steps 10

Smoke first.  A full 3-epoch run saves ``epoch_{1,2,3}/training_state.pt``
(fp32 master model weights; optimizer moments only with
``--save-optimizer-every-epoch``) and ``epoch_{1,2,3}/hf_model`` (bf16 export
for vLLM evaluation).  Resume is not implemented yet; run long jobs under tmux.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from torch.distributed.fsdp import (  # noqa: E402
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from e1_alignment.rebuild.training_math import (  # noqa: E402
    backward_window,
    lr_scale,
    supervised_tokens,
)
from e1_alignment.student_sft_data import (  # noqa: E402
    StudentDataset,
    collate,
    load_examples,
    masked_loss,
    summary as data_summary,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="per-rank micro-batches; 4 ranks x 2 = effective batch 8")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="full-FT LR; lower than the 5e-5 LoRA recipe because all "
                        "weights update and the base is already SFT-ed")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--smoke-save-model", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="smoke test also exports one full HF checkpoint")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="0 disables; run under tmux for long jobs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every-epoch", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--save-optimizer-every-epoch", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="also save AdamW moments (~2x model size per epoch); "
                        "off by default because resume is not implemented")
    p.add_argument("--save-training-state", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="save fp32 training_state.pt in epoch checkpoints; "
                        "disable for HF-only eval checkpoints")
    p.add_argument("--save-every-steps", type=int, default=0,
                   help="additional HF-only checkpoint every N optimizer steps "
                        "(0 disables; epoch-boundary steps are left to the epoch save)")
    p.add_argument("--min-free-disk-gb", type=float, default=20.0,
                   help="refuse to start if estimated checkpoint volume + this margin "
                        "exceeds free disk")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--fsdp-strategy", choices=("full", "hybrid", "shard_grad_op"),
                   default="full")
    p.add_argument("--wandb-mode", choices=("disabled", "offline", "online"),
                   default=None, help="default: $WANDB_MODE, else disabled")
    p.add_argument("--wandb-project", default=None,
                   help="default: $WANDB_PROJECT, else sqlrl")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-dir", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-only data/mask check; do not initialize FSDP")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def init_wandb(args, out, data_info, world, steps_per_epoch, total_steps, rank):
    """Initialise W&B on rank 0 only; never fail training because of logging."""
    if rank != 0:
        return None, {"mode": "non_rank0"}
    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "disabled")
    if mode == "disabled":
        return None, {"mode": "disabled"}
    try:
        import wandb
        project = args.wandb_project or os.environ.get("WANDB_PROJECT", "sqlrl")
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
        kwargs = {
            "project": project,
            "name": args.wandb_run_name or out.name,
            "dir": args.wandb_dir or str(out),
            "mode": mode,
            "config": {
                "args": vars(args),
                "data": data_info,
                "world_size": world,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
            },
        }
        if entity:
            kwargs["entity"] = entity
        if args.wandb_group:
            kwargs["group"] = args.wandb_group
        run = wandb.init(**kwargs)
        info = {
            "mode": mode,
            "project": project,
            "name": args.wandb_run_name or out.name,
            "run_id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
        }
        print(json.dumps({"wandb": info}, ensure_ascii=False), flush=True)
        return run, info
    except Exception as exc:
        info = {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({"wandb_init_failed": info}, ensure_ascii=False), flush=True)
        return None, info


def build_fsdp_model(args, local_rank):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    # Keep fp32 master weights.  Loading bf16 and letting AdamW update bf16
    # parameters quantises every update to ~1 ulp (~6e-5 near 0.02), which can
    # round an lr=2e-5/5e-5 step to zero.  FSDP mixed precision casts to bf16 for
    # forward/backward while the optimizer still sees fp32 parameters.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    if args.fsdp_strategy == "full":
        strategy = ShardingStrategy.FULL_SHARD
    elif args.fsdp_strategy == "hybrid":
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        strategy = ShardingStrategy.SHARD_GRAD_OP
    device = torch.device("npu", local_rank)
    model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            # Do not cast buffers: rotary inv_freq must stay fp32.
            buffer_dtype=None,
        ),
        auto_wrap_policy=wrap_policy,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
    )
    return model


def clean_full_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("_fsdp_wrapped_module.", "")
        key = key.replace("module.", "", 1)
        cleaned[key] = value
    return cleaned


def force_bf16_config(hf_dir):
    """Make a saved HF config advertise bf16 (transformers 4.57 uses ``dtype``)."""
    path = Path(hf_dir) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["dtype"] = "bfloat16"
    config.pop("torch_dtype", None)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def estimate_checkpoint_gb(model, args, world=1, total_steps=None):
    """Conservative estimate of all checkpoints written to disk.

    FSDP with ``use_orig_params=True`` exposes the *local* shard through
    ``model.parameters()`` (observed: 771M for a 3.086B model on 4 ranks), so
    multiply by the shard-group size to recover the full parameter count.
    Mid-run (``--save-every-steps``) checkpoints are HF-only.
    """
    local_params = sum(p.numel() for p in model.parameters())
    params = local_params * world
    fp32_gb = params * 4 / 1e9
    bf16_gb = params * 2 / 1e9
    per_epoch = (fp32_gb if args.save_training_state else 0.0) + bf16_gb
    if (args.save_optimizer_every_epoch and args.save_training_state
            and not args.smoke_steps):
        per_epoch += 2 * fp32_gb   # AdamW exp_avg + exp_avg_sq
    if args.smoke_steps:
        return per_epoch, params
    n_epoch = args.epochs if args.checkpoint_every_epoch else 0
    n_mid = (total_steps // args.save_every_steps
             if args.save_every_steps and total_steps else 0)
    return n_epoch * per_epoch + n_mid * bf16_gb, params


def save_checkpoint(model, optimizer, tokenizer, out, step, epoch, args,
                    save_optimizer=None, save_training_state=None, name=None):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    opt_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if save_training_state is None:
        save_training_state = args.save_training_state
    if save_optimizer is None:
        save_optimizer = args.save_optimizer_every_epoch
    if not save_training_state:
        save_optimizer = False
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, opt_cfg):
        model_sd = model.state_dict()
        optim_sd = (FSDP.optim_state_dict(model, optimizer)
                    if save_optimizer else None)
    ckpt_path = None
    if int(os.environ.get("RANK", "0")) == 0:
        ckpt = Path(out) / (name or f"epoch_{epoch}")
        ckpt.mkdir(parents=True, exist_ok=True)
        if save_training_state:
            torch.save({
                "model": model_sd,
                "optimizer": optim_sd,
                "step": step,
                "epoch": epoch,
                "config": vars(args),
            }, ckpt / "training_state.pt")
        (ckpt / "checkpoint_meta.json").write_text(
            json.dumps({
                "step": step, "epoch": epoch,
                "has_training_state": save_training_state,
                "has_optimizer": optim_sd is not None,
                "config": vars(args),
            }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        hf_dir = ckpt / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)
        # The trainer keeps fp32 masters, but checkpoints used for vLLM
        # inference are exported in bf16 (same dtype the evaluator requests).
        config = model.module.config
        saved_dtype = getattr(config, "dtype", None)
        saved_torch_dtype = getattr(config, "torch_dtype", None)
        saved_use_cache = config.use_cache
        config.dtype = torch.bfloat16
        try:
            config.torch_dtype = torch.bfloat16
        except Exception:
            pass  # deprecated alias; ``dtype`` is authoritative in 4.57+
        config.use_cache = True
        hf_state = {key: value.to(torch.bfloat16)
                    for key, value in clean_full_state_dict(model_sd).items()}
        model.module.save_pretrained(
            hf_dir, state_dict=hf_state, safe_serialization=True)
        tokenizer.save_pretrained(hf_dir)
        # transformers 4.57 serializes ``dtype`` (not ``torch_dtype``); enforce
        # it explicitly so HF/vLLM do not upcast the bf16 export to fp32.
        force_bf16_config(hf_dir)
        config.dtype = saved_dtype
        try:
            config.torch_dtype = saved_torch_dtype
        except Exception:
            pass
        config.use_cache = saved_use_cache
        del hf_state
        ckpt_path = str(ckpt)
        rank0_print(json.dumps({"checkpoint": ckpt_path, "step": step,
                                "epoch": epoch,
                                "has_training_state": save_training_state}),
                    flush=True)
    torch.distributed.barrier()
    return ckpt_path


def main():
    args = parse_args()
    if not (args.epochs >= 1 and args.grad_accum >= 1 and args.lr > 0
            and args.max_context > 0 and args.smoke_steps >= 0):
        raise SystemExit("invalid training configuration")
    if args.dry_run:
        rows, excluded = load_examples(args.data, max_context=args.max_context,
                                       require_trainable=True)
        info = data_summary(rows, excluded)
        info["data"] = args.data
        info["ok"] = bool(rows) and not excluded
        print(json.dumps(info, ensure_ascii=False, indent=1))
        if not info["ok"]:
            raise SystemExit(2)
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hccl")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    out = Path(args.out)
    # Only rank 0 may inspect/decide; broadcast avoids the race where rank 0
    # creates the directory before another rank reaches the exists() check.
    if rank == 0:
        out_exists = out.exists()
    else:
        out_exists = False
    exists_flag = torch.tensor(
        [1 if (rank == 0 and out_exists and not args.overwrite) else 0],
        device=device, dtype=torch.int32)
    torch.distributed.broadcast(exists_flag, src=0)
    if int(exists_flag.item()) == 1:
        raise SystemExit(f"{out} exists; pass --overwrite to replace")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    rows, excluded = load_examples(args.data, max_context=args.max_context,
                                   require_trainable=True)
    if not rows or excluded:
        raise SystemExit(f"dataset rejected: kept={len(rows)} excluded={len(excluded)}")
    data_info = data_summary(rows, excluded)
    data_info.update({
        "data": args.data,
        "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "max_context": args.max_context,
        "require_trainable": True,
    })
    rank0_print(json.dumps(data_info, ensure_ascii=False), flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StudentDataset(rows)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler,
                        collate_fn=collate, num_workers=0)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.smoke_steps:
        total_steps = args.smoke_steps

    started = time.monotonic()
    model = build_fsdp_model(args, local_rank)
    required_gb, n_params = estimate_checkpoint_gb(
        model, args, world, total_steps=total_steps)
    free_gb = shutil.disk_usage(out).free / 1e9
    rank0_print(json.dumps({
        "parameters": n_params,
        "checkpoint_estimate_gb": round(required_gb, 1),
        "free_disk_gb": round(free_gb, 1),
        "min_free_disk_margin_gb": args.min_free_disk_gb,
    }), flush=True)
    if free_gb < required_gb + args.min_free_disk_gb:
        raise SystemExit(
            f"insufficient disk: need {required_gb + args.min_free_disk_gb:.1f}GB, "
            f"free {free_gb:.1f}GB; disable optimizer checkpoints or free space")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: lr_scale(index, total_steps))

    rank0_print(json.dumps({
        "world_size": world, "local_rank": local_rank,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "grad_accum": args.grad_accum,
        "effective_batch": world * args.grad_accum,
        "trainable_parameters": (
            sum(p.numel() for p in model.parameters() if p.requires_grad) * world),
    }), flush=True)

    wandb_run, wandb_info = init_wandb(
        args, out, data_info, world, steps_per_epoch, total_steps, rank)
    checkpoints = []
    epoch_losses, epoch_grads = [], []

    torch_npu.npu.reset_peak_memory_stats()
    step = 0
    current_epoch = 0
    stop_reason = None
    last_saved_step = -1
    failure = None
    log = (out / "train_log.jsonl").open("w", encoding="utf-8") if rank == 0 else None
    try:
        model.train()
        for epoch in range(args.epochs):
            current_epoch = epoch + 1
            epoch_losses, epoch_grads = [], []
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            while True:
                if args.max_steps and step >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                if args.max_seconds and time.monotonic() - started > args.max_seconds:
                    stop_reason = "max_seconds"
                    break
                batches = list(islice(iterator, args.grad_accum))
                if not batches:
                    break
                local_tokens = sum(supervised_tokens(batch) for batch in batches)
                count_tensor = torch.tensor([float(local_tokens)], device=device,
                                            dtype=torch.float32)
                torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
                global_tokens = max(int(round(float(count_tensor.item()))), 1)
                loss_scale = world * local_tokens / global_tokens
                loss, token_count = backward_window(
                    model, batches, masked_loss, loss_scale=loss_scale)
                norm = float(model.clip_grad_norm_(1.0))
                if not math.isfinite(norm):
                    raise FloatingPointError(
                        f"nonfinite_grad_norm at step {step + 1}: {norm}")
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss) / max(loss_scale, 1e-12)
                if rank == 0:
                    epoch_losses.append(loss_value)
                    epoch_grads.append(norm)
                    record = {
                        "step": step, "epoch": epoch + 1,
                        "loss": round(loss_value, 6),
                        "loss_scale": round(loss_scale, 6),
                        "grad_norm": round(norm, 6),
                        "supervised_tokens_local": token_count,
                        "supervised_tokens_global": global_tokens,
                        "micro_batches": len(batches),
                        "lr": used_lr,
                        "elapsed_s": round(time.monotonic() - started, 1),
                        "hbm_peak_gb": round(
                            torch_npu.npu.max_memory_allocated() / 1e9, 3),
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/loss": loss_value,
                            "train/loss_scale": loss_scale,
                            "train/grad_norm": norm,
                            "train/lr": used_lr,
                            "train/supervised_tokens_local": token_count,
                            "train/supervised_tokens_global": global_tokens,
                            "train/micro_batches": len(batches),
                            "train/hbm_peak_gb": record["hbm_peak_gb"],
                            "train/elapsed_s": record["elapsed_s"],
                            "train/epoch": epoch + 1,
                        }, step=step)
                if (args.save_every_steps and step % args.save_every_steps == 0
                        and step % steps_per_epoch != 0):
                    ckpt = save_checkpoint(
                        model, optimizer, tokenizer, out, step, current_epoch,
                        args, save_training_state=False,
                        name=f"step_{step:04d}")
                    last_saved_step = step
                    if rank == 0 and ckpt:
                        checkpoints.append({
                            "name": f"step_{step:04d}", "step": step,
                            "epoch": current_epoch, "path": ckpt,
                            "has_training_state": False,
                        })
                        if wandb_run is not None:
                            wandb_run.log({"checkpoint/saved": 1}, step=step)
                if args.smoke_steps and step >= args.smoke_steps:
                    stop_reason = "smoke_steps"
                    break
            if stop_reason:
                break
            if rank == 0 and wandb_run is not None and epoch_losses:
                wandb_run.log({
                    "epoch/loss_mean": sum(epoch_losses) / len(epoch_losses),
                    "epoch/grad_mean": sum(epoch_grads) / len(epoch_grads),
                    "epoch/grad_max": max(epoch_grads),
                    "epoch/lr_end": optimizer.param_groups[0]["lr"],
                }, step=step)
            if (args.checkpoint_every_epoch and not args.smoke_steps
                    and last_saved_step != step):
                ckpt = save_checkpoint(
                    model, optimizer, tokenizer, out, step, current_epoch, args,
                    name=f"epoch_{current_epoch}")
                last_saved_step = step
                if rank == 0 and ckpt:
                    checkpoints.append({
                        "name": f"epoch_{current_epoch}", "step": step,
                        "epoch": current_epoch, "path": ckpt,
                        "has_training_state": args.save_training_state,
                    })
                    if wandb_run is not None:
                        wandb_run.log({"checkpoint/saved": 1}, step=step)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if log is not None:
            log.close()
        # Only save on a normal exit.  Saving from ``finally`` after one rank
        # failed would enter FSDP collectives with a rank that already left,
        # turning a training error into a hang.
        if failure is None and torch.distributed.is_initialized():
            if args.smoke_steps:
                if args.smoke_save_model and last_saved_step != step:
                    ckpt = save_checkpoint(model, optimizer, tokenizer, out, step,
                                           max(current_epoch, 1), args,
                                           save_optimizer=False)
                    last_saved_step = step
                    if rank == 0 and ckpt:
                        checkpoints.append({
                            "name": f"epoch_{max(current_epoch, 1)}",
                            "step": step, "epoch": max(current_epoch, 1),
                            "path": ckpt,
                            "has_training_state": args.save_training_state,
                        })
            elif last_saved_step != step:
                ckpt = save_checkpoint(model, optimizer, tokenizer, out, step,
                                       current_epoch, args)
                last_saved_step = step
                if rank == 0 and ckpt:
                    checkpoints.append({
                        "name": f"epoch_{current_epoch}", "step": step,
                        "epoch": current_epoch, "path": ckpt,
                        "has_training_state": args.save_training_state,
                    })
        elif rank == 0:
            print(json.dumps({"skipped_final_checkpoint": failure}), flush=True)
        if rank == 0:
            result = {
                "completed": failure is None and stop_reason in (None, "max_steps"),
                "failure": failure,
                "stop_reason": stop_reason,
                "steps": step,
                "total_steps": total_steps,
                "epochs": args.epochs,
                "world_size": world,
                "grad_accum": args.grad_accum,
                "effective_batch": world * args.grad_accum,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "fsdp_strategy": args.fsdp_strategy,
                "gradient_checkpointing": args.gradient_checkpointing,
                "save_optimizer_every_epoch": args.save_optimizer_every_epoch,
                "save_training_state": args.save_training_state,
                "save_every_steps": args.save_every_steps,
                "max_context": args.max_context,
                "model": str(Path(args.model).resolve()),
                "out": str(out),
                "data": data_info,
                "normalization": "causal supervised-token mean; FSDP averages rank gradients",
                "master_dtype": "float32",
                "compute_dtype": "bfloat16",
                "elapsed_s": round(time.monotonic() - started, 1),
                "hbm_peak_gb": round(torch_npu.npu.max_memory_allocated() / 1e9, 3),
                "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "wandb": wandb_info,
                "checkpoints": checkpoints,
            }
            (out / "train_summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if wandb_run is not None:
                try:
                    wandb_run.summary.update({
                        "completed": result["completed"],
                        "steps": step,
                        "elapsed_s": result["elapsed_s"],
                        "hbm_peak_gb": result["hbm_peak_gb"],
                        "data_sha256": data_info.get("sha256"),
                    })
                    wandb_run.finish(exit_code=0 if failure is None else 1)
                except Exception as exc:
                    print(json.dumps({"wandb_finish_failed": str(exc)}), flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
