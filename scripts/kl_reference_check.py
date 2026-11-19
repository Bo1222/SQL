"""Measure per-token KL between two HF checkpoints (base vs merged SFT).

Used by M6-05 to decide whether the observed actor/kl_loss ~0.5 is consistent
with "reference = base model" or with a correct merged reference.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "npu:0"


def load(path):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE).eval()
    return model


def token_logprobs(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), use_cache=False)
        logits = out.logits[0, :-1].float()
        logp = logits.log_softmax(-1)
        picked = logp.gather(1, ids[1:].unsqueeze(-1)).squeeze(-1)
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--other", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--samples", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", default="reports/m6/kl_reference_check.json")
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.other)
    rows = [json.loads(line) for line in Path(a.samples).read_text(encoding="utf-8").splitlines() if line.strip()][:a.n]

    base = load(a.base)
    other = load(a.other)
    results = []
    for row in rows:
        # prompt + a short deterministic continuation (the gold SQL in an answer tag)
        text = row["question"] + "\n<answer>" + row["gold_sql"][:400] + "</answer>"
        ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)[:1024], device=DEVICE)
        if ids.numel() < 8:
            continue
        lp_base = token_logprobs(base, ids)
        lp_other = token_logprobs(other, ids)
        delta = (lp_base - lp_other).float()  # ref - policy convention
        kl_k3 = (delta.exp() - delta - 1).mean().item()
        results.append({"id": row["id"], "tokens": int(ids.numel()),
                        "mean_abs_delta": float(delta.abs().mean().item()),
                        "kl_k3_per_token": kl_k3})
    summary = {
        "n": len(results),
        "kl_k3_per_token_mean": sum(r["kl_k3_per_token"] for r in results) / max(len(results), 1),
        "mean_abs_delta_mean": sum(r["mean_abs_delta"] for r in results) / max(len(results), 1),
        "per_sample": results,
        "interpretation": "per-token k3 KL between the two checkpoints on real BIRD text",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "kl_k3_per_token_mean", "mean_abs_delta_mean")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
