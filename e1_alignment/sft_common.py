"""Model/mask helpers shared by student-only SFT without the M6-08j import chain.

Importing ``e1_alignment.sft_train`` pulls in ``e1_alignment.rebuild.freeze``
and therefore ``sqlglot``.  The student-only trainer must not depend on the
M6-08j parser contract, so the two helpers it needs live here.
"""
from __future__ import annotations

import torch


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
    ids = sample["input_ids"][None, :].to(device)
    labels = sample["labels"][None, :].to(device)
    raw_mask = torch.tensor(dataset.rows[0]["loss_mask"], dtype=torch.bool)
    alignment_ok = bool(torch.equal(labels[0].cpu() != -100, raw_mask))
    if not alignment_ok:
        raise RuntimeError("labels do not align with the stored segment loss mask")
    attention = torch.ones_like(ids)
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
        masked_labels = labels.clone()
        masked_labels[labels == -100] = 0  # unmasked tokens become supervised in the control
        control = model(input_ids=ids, attention_mask=attention, labels=masked_labels,
                        use_cache=False).loss
    if not bool(torch.isfinite(base)) or not bool(torch.isfinite(control)):
        raise FloatingPointError("nonfinite_mask_check")
    supervised = int((labels[..., 1:] != -100).sum())
    total = int(labels.numel())
    model.train()
    return {
        "supervised_tokens": supervised,
        "total_tokens": total,
        "masked_loss": float(base),
        "unmasked_control_loss": float(control),
        "mask_changes_loss": abs(float(base) - float(control)) > 1e-6,
        "labels_align_with_segment_mask": alignment_ok,
    }
