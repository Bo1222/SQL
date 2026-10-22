"""BIRD SQL/domain SFT or observation-masked tool SFT; CPU/NPU, bounded smoke support."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agent.common import choose_device, load_model, read_jsonl, save_adapter, tokenizer_fingerprint, write_json
from agent.trajectory import Trajectory, masked_logprob
from train.grpo_tracking import log_metrics, training_logs
from train.project_env import load_project_env


def prepare_examples(rows, tokenizer, max_context=6144, stats=None):
    fingerprint = tokenizer_fingerprint(tokenizer)
    examples, excluded = [], []
    for row in rows:
        if row.get("split") != "sft":
            raise ValueError("SFT only accepts split=sft; validation/test/RL data must not enter cold start")
        if "trajectory" in row:
            if row.get("tokenizer_sha256") != fingerprint:
                raise ValueError("Trajectory tokenizer differs; regenerate using the training tokenizer")
            trajectory = Trajectory.from_dict(row["trajectory"])
            ids, mask = trajectory.input_ids, trajectory.loss_mask
        else:
            prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
            answer = tokenizer.encode("<answer>" + row["gold_sql"].strip() + "</answer>", add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                answer.append(tokenizer.eos_token_id)
            if len(prefix) + len(answer) > max_context and row.get("metadata_path"):
                # Large BIRD schemas use question-based retrieval, never gold-based table selection.
                from agent.common import load_metadata
                from agent.loop import AgentConfig
                from agent.protocol import build_prompt
                from agent.tools import DatabaseTools
                from eval.eval_agent import static_retrieval
                tools = DatabaseTools(row, load_metadata(row))
                config = AgentConfig(max_context=max_context, max_prompt=min(1536, max_context - 1))
                context = static_retrieval(row, tokenizer, tools, config)
                prompt = (build_prompt(row, tools.catalog(), 0) + "\n### Fixed retrieval\n" + context
                          + "\nNo tool calls are available. Return only <answer>final SQL</answer>.\n")
                prefix = tokenizer.encode(prompt, add_special_tokens=False)
                if stats is not None:
                    stats["question_retrieved_large_schemas"] = stats.get("question_retrieved_large_schemas", 0) + 1
            ids, mask = prefix + answer, [0] * len(prefix) + [1] * len(answer)
        if len(ids) > max_context or not any(mask):
            excluded.append({"id": row["id"], "tokens": len(ids), "reason": "over_budget_or_no_actions"})
        else:
            examples.append((ids, mask))
    if not examples:
        raise ValueError("No usable SFT trajectories")
    return examples, excluded


def collate_examples(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, (values, mask) in enumerate(batch):
        ids[i, :len(values)] = torch.tensor(values)
        attention[i, :len(values)] = 1
        labels[i, :len(values)] = torch.where(torch.tensor(mask, dtype=torch.bool), torch.tensor(values), -100)
    return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def main():
    load_project_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default="checkpoints/sft_lora")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "npu"], default="auto")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default=os.environ.get("WANDB_MODE", "offline"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sqlrl-agent"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-dir", default=None)
    a = p.parse_args()
    if min(a.epochs, a.grad_accum, a.max_context) < 1 or a.lr <= 0 or a.max_steps < 0:
        p.error("invalid training budgets")
    if (Path(a.out) / "adapter_config.json").exists():
        p.error("output already contains an adapter; choose a new output directory")
    torch.manual_seed(a.seed)
    device = choose_device(a.device)
    model, tok = load_model(a.model, a.adapter, device, trainable=True)
    preparation = {}
    examples, excluded = prepare_examples(read_jsonl(a.data), tok, a.max_context, preparation)
    write_json(Path(a.out) / "excluded.json", excluded)
    write_json(Path(a.out) / "data_preparation.json", preparation)
    with training_logs(a, 0, 1) as (log_file, run):
        train(a, model, tok, examples, excluded, log_file, run)


def train(a, model, tok, examples, excluded, log_file, run):
    loader = DataLoader(examples, batch_size=1, shuffle=True, collate_fn=lambda b: b)
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.)
    started, step = time.monotonic(), 0
    total = min(a.max_steps or math.inf, math.ceil(len(examples) / a.grad_accum) * a.epochs)
    for epoch in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        window, loss_sum = 0, 0.
        for index, [(ids, mask)] in enumerate(loader):
            model.train()
            loss = -masked_logprob(model, ids, mask, "policy", True).mean()
            loss.backward()
            window += 1
            loss_sum += float(loss.detach())
            if window < a.grad_accum and index + 1 < len(loader):
                continue
            for param in params:
                if param.grad is not None:
                    param.grad.div_(window)  # final partial window uses its actual size
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "loss": loss_sum / window, "grad_norm": norm, "lr": a.lr,
                      "elapsed_s": time.monotonic() - started}
            log_metrics(log_file, run, record)
            print(json.dumps(record), flush=True)
            window, loss_sum = 0, 0.
            if step >= total:
                break
        if step >= total:
            break
    save_adapter(model, tok, a.out)
    summary = {**vars(a), "steps": step, "kept": len(examples), "excluded": len(excluded),
               "elapsed_s": time.monotonic() - started, "tokenizer_sha256": tokenizer_fingerprint(tok)}
    write_json(Path(a.out) / "train_summary.json", summary)
    if run is not None:
        run.summary.update(summary)


if __name__ == "__main__":
    main()
