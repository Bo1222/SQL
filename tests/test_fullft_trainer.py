"""CPU-only checks for the FSDP full-parameter trainer helpers."""
from types import SimpleNamespace

import torch

from e1_alignment.student_sft_full import (
    clean_full_state_dict,
    estimate_checkpoint_gb,
    force_bf16_config,
)

PARAMS = 10_000_000


class Tiny(torch.nn.Module):
    def __init__(self, n=PARAMS):
        super().__init__()
        # Only numel() is consumed by the estimator.
        self.w = torch.nn.Parameter(torch.zeros(n, dtype=torch.float32))


def args(**overrides):
    base = {"epochs": 3, "smoke_steps": 0,
            "save_optimizer_every_epoch": False,
            "save_training_state": True,
            "checkpoint_every_epoch": True,
            "save_every_steps": 0}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_checkpoint_estimate_without_optimizer():
    # 4 bytes fp32 + 2 bytes bf16 = 6 bytes/param/epoch; 3 epochs.
    gb, params = estimate_checkpoint_gb(Tiny(), args())
    assert params == PARAMS
    assert abs(gb - PARAMS * 6 * 3 / 1e9) < 1e-6


def test_checkpoint_estimate_with_optimizer_and_smoke():
    gb, _ = estimate_checkpoint_gb(
        Tiny(), args(save_optimizer_every_epoch=True))
    # + 2 * fp32 moments per epoch: 4 + 2 + 8 bytes/param/epoch.
    assert abs(gb - PARAMS * 14 * 3 / 1e9) < 1e-6
    smoke_gb, _ = estimate_checkpoint_gb(
        Tiny(), args(smoke_steps=10, save_optimizer_every_epoch=True))
    # Smoke writes only one epoch and no optimizer.
    assert abs(smoke_gb - PARAMS * 6 / 1e9) < 1e-6


def test_checkpoint_estimate_scales_with_fsdp_world():
    # FSDP exposes local shards; the estimator must multiply by world size.
    gb1, params1 = estimate_checkpoint_gb(Tiny(), args(), world=1)
    gb4, params4 = estimate_checkpoint_gb(Tiny(), args(), world=4)
    assert params4 == 4 * params1
    assert abs(gb4 - 4 * gb1) < 1e-6


def test_checkpoint_estimate_hf_only_mid_checkpoints():
    gb, _ = estimate_checkpoint_gb(
        Tiny(),
        args(save_training_state=False, checkpoint_every_epoch=False,
             save_every_steps=197),
        world=1, total_steps=1182)
    # 1182 // 197 = 6 HF-only checkpoints, 2 bytes/param each.
    assert abs(gb - PARAMS * 2 * 6 / 1e9) < 1e-6


def test_clean_full_state_dict_strips_fsdp_wrappers():
    state = {
        "_fsdp_wrapped_module.model.embed_tokens.weight": 1,
        "_fsdp_wrapped_module.model.layers.0._fsdp_wrapped_module."
        "self_attn.q_proj.weight": 2,
    }
    cleaned = clean_full_state_dict(state)
    assert set(cleaned) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    }


def test_hf_export_roundtrip_with_tied_embeddings(tmp_path):
    """The export path must save a reloadable bf16 HF checkpoint."""
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    prefixed = {"_fsdp_wrapped_module." + key: value
                for key, value in model.state_dict().items()}
    cleaned = clean_full_state_dict(prefixed)
    assert "model.embed_tokens.weight" in cleaned
    hf_state = {key: value.to(torch.bfloat16) for key, value in cleaned.items()}
    model.config.torch_dtype = torch.bfloat16
    model.config.use_cache = True
    model.save_pretrained(tmp_path, state_dict=hf_state, safe_serialization=True)
    force_bf16_config(tmp_path)

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert reloaded.config.dtype == torch.bfloat16
    assert reloaded.config.use_cache is True
    assert reloaded.config.tie_word_embeddings is True
    assert reloaded.lm_head.weight.data_ptr() == \
        reloaded.model.embed_tokens.weight.data_ptr()
