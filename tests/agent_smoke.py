"""Bounded integration smoke: real tiny Qwen/LoRA + scripted actions + real SQL.

Only generation is controlled. Token scoring, checkpoint recomputation, optimizer,
distributed gradient reduction, adapter persistence, tools and W&B are real.
"""
import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path


def prepare(root):
    import torch
    from agent.common import load_model, save_adapter, write_json, write_jsonl
    from data.prepare_bird_agent import build_metadata
    from tests.test_grpo_utils import byte_tokenizer, tiny_model
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    tok = byte_tokenizer()
    tok.save_pretrained(root / "model")
    tiny_model(len(tok)).save_pretrained(root / "model")
    model, tok = load_model(str(root / "model"), "", "cpu", trainable=True)
    save_adapter(model, tok, root / "initial_adapter")
    rows = []
    for name in ["numbers", "heldout"]:
        path = root / f"{name}.sqlite"
        if not path.exists():
            with sqlite3.connect(path) as con:
                con.execute("CREATE TABLE numbers(n INTEGER)")
                con.executemany("INSERT INTO numbers VALUES (?)", [(1,), (2,)])
        metadata = root / f"{name}.json"
        write_json(metadata, build_metadata(path))
        for index in range(2):
            rows.append({"id": f"{name}-{index}", "db_id": name, "db_path": str(path.resolve()),
                         "metadata_path": str(metadata.resolve()), "question": f"How many numbers? Example {index}.",
                         "evidence": "", "gold_sql": "SELECT count(*) FROM numbers",
                         "prompt": "CREATE TABLE numbers(n INTEGER);\nHow many numbers?\n",
                         "split": "val" if name == "heldout" else "rl"})
    write_jsonl(root / "rl.jsonl", [r for r in rows if r["split"] == "rl"])
    write_jsonl(root / "val.jsonl", [r for r in rows if r["split"] == "val"])
    write_jsonl(root / "sft.jsonl", [{**r, "split": "sft"} for r in rows if r["split"] == "rl"])


def worker(root):
    import sys
    import torch
    import train.agent_grpo_ddp as trainer
    from agent.common import write_json
    real_loader = trainer.load_model
    captured = {}
    def loader(*args, **kwargs):
        model, tok = real_loader(*args, **kwargs)
        captured["model"] = model
        captured["before"] = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
        return model, tok
    class ControlledGenerator:
        def __init__(self, model, tokenizer, temperature):
            self.tok, self.temperature, self.episodes = tokenizer, temperature, 0
            self.good = True
        def __call__(self, ids, max_new_tokens):
            text = self.tok.decode(ids, skip_special_tokens=True).split("### Trajectory\n", 1)[-1]
            if "</observation>" not in text:
                self.good = self.temperature == 0 or self.episodes % 2 == 0
                self.episodes += 1
                action = "<sql>SELECT missing FROM numbers</sql>"
            else:
                query = "SELECT count(*) FROM numbers" if self.good else "SELECT count(*) + 1 FROM numbers"
                action = "<answer>" + query + "</answer>"
            values = self.tok.encode(action, add_special_tokens=False)
            assert len(values) <= max_new_tokens
            return values
    trainer.load_model, trainer.HFGenerator = loader, ControlledGenerator
    # Original CLI arguments, supplied by the smoke command, are used unchanged.
    trainer.main()
    after = dict(captured["model"].named_parameters())
    ref = [n for n in captured["before"] if ".ref." in n]
    policy = [n for n in captured["before"] if ".policy." in n]
    result = {"rank": int(os.environ.get("RANK", 0)), "reference_parameters": len(ref),
              "reference_unchanged": bool(ref) and all(torch.equal(captured["before"][n], after[n].detach().cpu()) for n in ref),
              "policy_changed": any(not torch.equal(captured["before"][n], after[n].detach().cpu()) for n in policy)}
    write_json(root / f"rank_{result['rank']}_checks.json", result)
    assert result["reference_unchanged"] and result["policy_changed"], result


def long_context(root):
    import time
    import torch
    from agent.common import choose_device, load_model, write_json
    from agent.trajectory import masked_logprob
    started = time.monotonic()
    model, tok = load_model("/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct",
                            "checkpoints/sft_lora", choose_device("npu"), trainable=True, reference=True)
    prefix = tok.encode("You are a SQLite agent.\n", add_special_tokens=False)
    action_fragment = tok.encode("<sql>SELECT count(*) FROM numbers WHERE n > 0</sql>\n", add_special_tokens=False)
    # Upper-bound training-memory stress: maximum scored token budget, not just a short final SQL.
    action = (action_fragment * (2048 // len(action_fragment) + 1))[:2048]
    observation = tok.encode("<observation>Database value preview: monthly issuance, numbers 1 2 3.</observation>\n", add_special_tokens=False)
    middle_length = 6144 - len(prefix) - len(action)
    ids = prefix + (observation * (middle_length // len(observation) + 1))[:middle_length] + action
    mask = [0] * (len(ids) - len(action)) + [1] * len(action)
    ref_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".ref." in n}
    model.set_adapter("policy")
    params = [p for n, p in model.named_parameters() if ".policy." in n and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-6)
    model.train()
    loss = -masked_logprob(model, ids, mask, "policy", True).mean()
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    optimizer.step()
    ref_after = dict(model.named_parameters())
    result = {"model": "Qwen2.5-3B-Instruct", "context_tokens": len(ids), "action_tokens": sum(mask),
              "loss": float(loss.detach()), "grad_norm": norm,
              "reference_unchanged": all(torch.equal(p, ref_after[n].detach().cpu()) for n, p in ref_before.items()),
              "peak_mem_gb": torch.npu.max_memory_allocated() / 1e9,
              "elapsed_s": time.monotonic() - started,
              "scope": "one synthetic long-context supervised backward; not an accuracy experiment"}
    write_json(root / "real_3b_long_context_2048_actions.json", result)
    assert result["reference_unchanged"] and norm > 0
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys
    operation = sys.argv.pop(1)
    root = Path(sys.argv.pop(1)).resolve()
    if operation == "prepare":
        prepare(root)
    elif operation == "worker":
        worker(root)
    elif operation == "long":
        long_context(root)
    else:
        raise ValueError(operation)
