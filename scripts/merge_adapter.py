"""Merge a PEFT LoRA adapter into the base model for vLLM offline inference."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="checkpoints/bird_agent_full_sft_20261026/seed_0/step_600")
    p.add_argument("--tokenizer-from", default="", help="dir with tokenizer files (defaults to --adapter)")
    p.add_argument("--out", default="artifacts/checkpoints/step_600_merged")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = a.tokenizer_from or a.adapter
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
