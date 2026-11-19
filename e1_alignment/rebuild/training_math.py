"""Causal supervised-token mean over a whole accumulation window."""
import math
import torch


def supervised_tokens(batch):
    # Causal LM loss predicts labels[1:], never labels[0].
    return int((batch["labels"][..., 1:] != -100).sum())


def backward_window(model, batches, loss_fn, before_micro=None, loss_scale=1.0):
    """Backward a token-weighted mean over one accumulation window.

    ``loss_scale`` lets a distributed caller correct for FSDP's equal-rank
    gradient averaging: each rank multiplies its local token mean by
    ``world_size * local_tokens / global_tokens`` before backward, so the
    averaged gradient equals the global supervised-token mean.
    """
    counts = [supervised_tokens(batch) for batch in batches]
    total = sum(counts)
    if total <= 0 or any(n <= 0 for n in counts):
        raise ValueError("empty_supervision_in_window")
    total_loss = 0.0
    for batch, count in zip(batches, counts):
        if before_micro:
            before_micro()
        loss = loss_fn(model, batch)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite_or_nonscalar_loss")
        weight = (count / total) * float(loss_scale)
        (loss * weight).backward()
        total_loss += float(loss.detach()) * weight
    return total_loss, total


def lr_scale(step, total_steps, warmup_fraction=.05):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
