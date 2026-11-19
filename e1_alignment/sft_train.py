"""E1-06: segment-masked LoRA SFT V2, initialized from the frozen SFT capability."""
import argparse
import hashlib
import json
import math
import signal
from itertools import islice
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from e1_alignment.common import read_jsonl, write_json
from e1_alignment.rebuild.training_math import backward_window, lr_scale
from e1_alignment.rebuild.freeze import verify_frozen


def load_examples(path, max_context):
    rows, excluded = [], []
    for row in read_jsonl(path):
        ids = list(row.get("input_ids") or [])
        mask = list(row.get("loss_mask") or [])
        if len(ids) != len(mask) or not ids:
            excluded.append({"id": row.get("id"), "reason": "ids_mask_mismatch"})
            continue
        if len(ids) > max_context:
            excluded.append({"id": row.get("id"), "reason": "over_budget", "tokens": len(ids)})
            continue
        if any(m not in (0, 1) for m in mask) or mask[0] or not any(mask[1:]):
            excluded.append({"id": row.get("id"), "reason": "no_supervised_tokens"})
            continue
        rows.append({"id": row.get("id"), "input_ids": ids, "loss_mask": mask})
    return rows, excluded


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["loss_mask"], dtype=torch.bool)
        labels = torch.where(mask, ids, torch.full_like(ids, -100))
        return {"input_ids": ids, "labels": labels, "ids": row["id"]}


def collate(batch):
    assert len(batch) == 1, "micro batch size is fixed at 1 for this NPU budget"
    item = batch[0]
    return {"input_ids": item["input_ids"][None, :],
            "attention_mask": torch.ones_like(item["input_ids"])[None, :],
            "labels": item["labels"][None, :], "id": item["ids"]}


def masked_loss(model, batch):
    device = next(model.parameters()).device
    output = model(input_ids=batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device),
                   labels=batch["labels"].to(device), use_cache=False)
    return output.loss


def build_model(model_path, lora_rank=16, lora_alpha=32, lora_dropout=0.05, device="npu:0"):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    # PEFT default initialization: random A, zero B => zero adapter output.
    b_weights = [p for name, p in model.named_parameters() if "lora_B" in name]
    if not b_weights or any(bool(torch.count_nonzero(p)) for p in b_weights):
        raise RuntimeError("new_adapter_is_not_zero_output_initialized")
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tokenizer


def mask_self_check(model, dataset, device):
    """Prove that non-masked tokens cannot contribute to the loss."""
    model.eval()
    sample = dataset[0]
    ids, labels = sample["input_ids"][None, :].to(device), sample["labels"][None, :].to(device)
    raw_mask = torch.tensor(dataset.rows[0]["loss_mask"], dtype=torch.bool)
    alignment_ok = bool(torch.equal(labels[0].cpu() != -100, raw_mask))
    if not alignment_ok:
        raise RuntimeError("labels do not align with the stored segment loss mask")
    attention = torch.ones_like(ids)
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
        masked_labels = labels.clone()
        masked_labels[labels == -100] = 0  # unmasked tokens become supervised in the control
        control = model(input_ids=ids, attention_mask=attention, labels=masked_labels, use_cache=False).loss
    if not bool(torch.isfinite(base)) or not bool(torch.isfinite(control)):
        raise FloatingPointError("nonfinite_mask_check")
    supervised = int((labels[..., 1:] != -100).sum())
    total = int(labels.numel())
    model.train()
    return {"supervised_tokens": supervised, "total_tokens": total,
            "masked_loss": float(base), "unmasked_control_loss": float(control),
            "mask_changes_loss": abs(float(base) - float(control)) > 1e-6,
            "labels_align_with_segment_mask": alignment_ok}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/m6/e1_alignment/e1_alignment_sft.jsonl")
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--out", required=True)
    p.add_argument("--quality-manifest", required=True,
                   help="M6-08j frozen data and individual-review hashes; mandatory before model load")
    p.add_argument("--max-seconds", type=float, default=3300)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--check-mask", action="store_true")
    a = p.parse_args()

    if a.epochs != 1 or a.grad_accum != 8 or a.lr != 2e-6 or a.max_seconds > 3300 or a.max_seconds <= 0 or a.max_context != 6144 or a.seed != 0:
        raise SystemExit("M6-08j fixed training contract mismatch")
    if (a.lora_rank, a.lora_alpha, a.lora_dropout) != (16, 32, .05):
        raise SystemExit("M6-08j LoRA contract mismatch")
    if Path(a.model).resolve() != Path("artifacts/checkpoints/step_600_merged").resolve():
        raise SystemExit("M6-08j must initialize from source merged SFT")
    if Path(a.out).exists():
        raise SystemExit("candidate output already exists; no overwrite or automatic repeat")
    from e1_alignment.rebuild.contract import verify_contract
    verify_contract()
    frozen = verify_frozen(a.data, a.quality_manifest)
    integrity_files = [Path(a.data), Path(a.quality_manifest), Path(frozen["reviews_path"]), Path(frozen["monitor_path"])]
    integrity_files += [Path(p) for p in frozen["isolation_input_hashes"]]
    integrity_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in integrity_files}
    if a.max_steps or a.smoke_steps:
        raise SystemExit("formal candidate must finish the single epoch")
    from e1_alignment.rebuild.budget import active_budget
    remaining_job_seconds = active_budget("train")
    started = time.monotonic()  # includes initialization and mask check
    class TrainingStopped(RuntimeError):
        pass
    def expired(signum, frame):
        raise TrainingStopped("training_hard_timeout")
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(a.max_seconds, remaining_job_seconds))

    import numpy as np
    import torch_npu  # noqa: F401

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    rows, excluded = load_examples(a.data, a.max_context)
    if not rows or excluded:
        raise SystemExit("frozen dataset has unusable examples; reject entire dataset")
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
    total_steps = (a.smoke_steps or a.max_steps or steps_per_epoch * a.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: lr_scale(index, total_steps))
    step = 0
    log_path = out / "train_log.jsonl"
    stop_reason = None
    def check_immutable_and_time():
        if time.monotonic() - started >= a.max_seconds:
            raise TrainingStopped("training_hard_timeout")
        if hashlib.sha256(Path(a.data).read_bytes()).hexdigest() != digest:
            raise TrainingStopped("dataset_hash_changed")
        for path in integrity_files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != integrity_hashes[str(path)]:
                raise TrainingStopped("frozen_artifact_hash_changed:" + str(path))
    try:
        with log_path.open("w", encoding="utf-8") as log:
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(a.epochs):
                iterator = iter(loader)
                while True:
                    batches = list(islice(iterator, a.grad_accum))
                    if not batches:
                        break
                    loss, token_count = backward_window(model, batches, masked_loss,
                                                         before_micro=check_immutable_and_time)
                    norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
                    used_lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = {"step": step, "epoch": epoch + 1, "loss": loss,
                              "grad_norm": norm, "supervised_tokens": token_count,
                              "micro_batches": len(batches), "lr": used_lr,
                              "elapsed_s": round(time.monotonic() - started, 1)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
            check_immutable_and_time()
    except (TrainingStopped, FloatingPointError, RuntimeError, ValueError) as exc:
        stop_reason = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    # Interrupted artifacts never occupy the final candidate path.
    save_path = out / "interrupted" if stop_reason else out
    model.save_pretrained(save_path, selected_adapters=["default"])
    tokenizer.save_pretrained(save_path)
    if stop_reason:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(), "step": step, "stop_reason": stop_reason},
                   save_path / "training_state.pt")

    summary = {
        "stage": "M6-08j", "completed": stop_reason is None, "stop_reason": stop_reason,
        "quality_manifest": a.quality_manifest, "quality_rule": frozen["rule_version"],
        "normalization": "causal supervised token mean over actual accumulation window",
        "schedule": "5% warmup, linear decay", "data": a.data, "data_sha256": digest,
        "kept": len(rows), "excluded": len(excluded),
        "model": a.model, "initialization": "merged SFT capability + fresh zero-init LoRA",
        "epochs": a.epochs, "steps": step, "grad_accum": a.grad_accum, "lr": a.lr,
        "max_context": a.max_context, "lora": {"r": a.lora_rank, "alpha": a.lora_alpha,
                                               "dropout": a.lora_dropout},
        "elapsed_s": round(time.monotonic() - started, 1),
        "mask_check": check, "train_log": str(log_path),
    }
    write_json(out / "train_summary.json", summary)
    write_json(out / "excluded.json", excluded)
    if stop_reason:
        raise SystemExit("training stopped: " + stop_reason)
    print(json.dumps({"done": True, **{k: summary[k] for k in ("steps", "elapsed_s")}}))


if __name__ == "__main__":
    main()
