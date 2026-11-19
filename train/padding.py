"""Remove shared padding columns before actor/ref computation for SQL RL.

The rollout still has the declared prompt/response limits. Only columns that
are padding in every selected trajectory are removed; positions, rewards and
action masks retain the same token alignment.
"""
import torch

RESPONSE_FIELDS = {
    "responses", "response_mask", "rollout_log_probs", "old_log_probs",
    "ref_log_prob", "token_level_scores", "token_level_rewards", "rm_scores",
    "advantages", "returns", "values", "entropys", "rollout_is_weights",
}


def compact_padding(batch):
    tensors = batch.batch
    response = tensors["responses"].shape[-1]
    total = tensors["input_ids"].shape[-1]
    prompt = total - response
    attention = tensors["attention_mask"]
    left, right = attention[:, :prompt], attention[:, prompt:]
    if torch.any(left[:, 1:].long() < left[:, :-1].long()) or torch.any(right[:, 1:].long() > right[:, :-1].long()):
        raise ValueError("expected left-padded prompts and right-padded responses")
    keep_prompt = max(1, int(left.sum(-1).max().item()))
    keep_response = max(1, int(right.sum(-1).max().item()))
    for key in ("response_mask", "token_level_scores", "rm_scores"):
        if key in tensors and torch.any(tensors[key][..., keep_response:] != 0):
            raise ValueError(f"nonzero {key} outside the response attention mask")
    for key in ("input_ids", "attention_mask", "position_ids"):
        value = tensors.get(key)
        if value is not None:
            tensors[key] = value[..., prompt - keep_prompt:prompt + keep_response]
    if "prompts" in tensors:
        tensors["prompts"] = tensors["prompts"][..., -keep_prompt:]
    for key in RESPONSE_FIELDS & set(tensors.keys()):
        tensors[key] = tensors[key][..., :keep_response]
    return {"sampling/actor_prompt_width": keep_prompt,
            "sampling/actor_response_width": keep_response,
            "sampling/removed_padding_columns": total - keep_prompt - keep_response}


def budget_clipping_metrics(batch, prompt_limit, response_limit):
    """Clipping is measured against declared limits, not compact tensor width."""
    width = batch.batch["responses"].shape[-1]
    attention = batch.batch["attention_mask"]
    prompt_lengths = attention[:, :-width].sum(-1)
    response_lengths = attention[:, -width:].sum(-1)
    active = response_lengths > 0
    return {
        "prompt_length/clip_ratio": float((prompt_lengths >= prompt_limit).float().mean().item()),
        "response_length/clip_ratio": float((response_lengths >= response_limit).float().mean().item()),
        "response_length_non_aborted/clip_ratio": float(
            (response_lengths[active] >= response_limit).float().mean().item()) if active.any() else 0.0,
    }
