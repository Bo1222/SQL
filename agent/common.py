"""Shared IO and model loading for the new pipeline; existing v1 stays independent."""
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def tokenizer_fingerprint(tokenizer):
    content = json.dumps({"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def load_metadata(row):
    return json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))


def select_rows(rows, limit=0, seed=0):
    """Deterministic proportional DB sampling for fast evaluation, never a sorted prefix."""
    if limit <= 0 or limit >= len(rows):
        return rows
    import random
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[row["db_id"]].append(row)
    quotas = {db: int(len(v) * limit / len(rows)) for db, v in groups.items()}
    remainder = sorted(groups, key=lambda db: (-(len(groups[db]) * limit / len(rows) - quotas[db]), db))
    for db in remainder[:limit - sum(quotas.values())]:
        quotas[db] += 1
    rng = random.Random(seed)
    result = []
    for db in sorted(groups):
        result.extend(rng.sample(sorted(groups[db], key=lambda r: str(r["id"])), quotas[db]))
    return result


def choose_device(device="auto", local_rank=0):
    import torch
    if device == "cpu":
        return "cpu"
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            torch.npu.set_device(local_rank)
            return f"npu:{local_rank}"
    except ImportError:
        pass
    if device == "npu":
        raise RuntimeError("NPU requested but unavailable")
    return "cpu"


def load_model(model_path, adapter="", device="cpu", trainable=False, reference=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16 if device.startswith("npu") else torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, adapter_name="policy", is_trainable=trainable)
        if reference:
            model.load_adapter(adapter, adapter_name="ref", is_trainable=False)
    elif trainable:
        if reference:
            raise ValueError("RL requires an explicit cold-start adapter for a frozen reference")
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"), adapter_name="policy")
    else:
        model = base
    model.to(device)
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def save_adapter(model, tokenizer, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.set_adapter("policy")
    model.save_pretrained(out, selected_adapters=["policy"])
    sub = out / "policy"
    if sub.is_dir():
        for path in sub.iterdir():
            path.replace(out / path.name)
        sub.rmdir()
    tokenizer.save_pretrained(out)
