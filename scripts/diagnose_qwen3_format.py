"""Root-cause diagnostic for Qwen3-4B raw-E1 format/stop failure.

This is not training and not a capability evaluation. It answers:
  1. Can the SFT model reproduce a *training* action under teacher forcing?
     (If yes, the problem is generation/decoding; if no, optimization/data.)
  2. Does the adapter change the base model's action-token behavior?
  3. On held-out dev200 prompts, is the first action's content valid even when
     the turn boundary is wrong (extra actions / no stop)?

For each training example we take the first *supervised* action segment, use all
previous segments as prefix, and compare base vs merged-SFT:
  - teacher-forced loss / token accuracy on the target action tokens;
  - greedy generation: number of complete actions, first-action validity,
    raw-SQL vs JSON-wrapped body, correct closing tag.

The training-prefix part is a memorization check, not a generalization test.
"""
import argparse
import gc
import json
import re
from pathlib import Path

import torch

ACTION_RE = re.compile(r"<(schema_lookup|sql_execute|answer)>(.*?)</\1>", re.S)
ACTION_OPEN_RE = re.compile(r"<(schema_lookup|sql_execute|answer)>")


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def load_model(path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)
    model.eval()
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    try:
        import torch_npu
        torch_npu.npu.empty_cache()
    except Exception:
        pass


def analyze_action(text):
    """Return format facts for the first action in a generated string."""
    text = text or ""
    complete = list(ACTION_RE.finditer(text))
    opens = list(ACTION_OPEN_RE.finditer(text))
    first = complete[0] if complete else None
    info = {
        "open_actions": len(opens),
        "complete_actions": len(complete),
        "has_first_action": first is not None,
        "one_complete_action": len(complete) == 1 and len(opens) == 1,
        "one_action_and_no_extra_text": False,
        "first_tag": None,
        "first_body_format": None,
        "first_action_valid": False,
        "first_action_text": None,
    }
    if not first:
        return info
    tag, body = first.group(1), first.group(2).strip()
    info["first_tag"] = tag
    info["first_action_text"] = first.group(0)
    info["one_action_and_no_extra_text"] = (
        len(complete) == 1 and len(opens) == 1 and text.strip() == first.group(0).strip())
    if tag in ("sql_execute", "answer"):
        if body.upper().startswith(("SELECT", "WITH")):
            fmt = "raw_sql"
        elif body.startswith("{"):
            fmt = "json_wrapped"
        else:
            fmt = "other"
    else:
        try:
            json.loads(body)
            fmt = "json_lookup"
        except Exception:
            fmt = "bad_json"
    info["first_body_format"] = fmt
    info["first_action_valid"] = (
        info["one_complete_action"] and fmt in ("raw_sql", "json_lookup")
        and "<observation>" not in first.group(0))
    return info


def teacher_forced(model, tokenizer, prefix_text, target_text, device):
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    input_ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prefix_ids) + target_ids],
                          dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    labels=labels, use_cache=False)
        logits = out.logits[:, len(prefix_ids) - 1:-1, :]
        target = labels[:, len(prefix_ids):]
        loss = float(out.loss.detach())
        pred = logits.argmax(-1)
        acc = float((pred == target).float().mean()) if target.numel() else 0.0
        first_acc = float((pred[:, 0] == target[:, 0]).float().mean()) if target.numel() else 0.0
    return {
        "prefix_tokens": len(prefix_ids),
        "target_tokens": len(target_ids),
        "loss": loss,
        "token_acc": acc,
        "first_token_acc": first_acc,
    }


def generate(model, tokenizer, prompt_text, device, max_new_tokens):
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0, len(ids):].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    info = analyze_action(text)
    info["generated_text"] = text[:2000]
    info["generated_tokens"] = len(gen_ids)
    return info


def pick_training_examples(rows, tokenizer, n, max_prefix_tokens=3500):
    """Pick up to n records with a supervised action, balanced by success_type."""
    picked, seen = [], set()
    wanted = ["autonomous_repair", "direct_sql_success", "gold_assisted",
              "lookup_success", "self_correction"]
    for success in wanted:
        for row in rows:
            if row.get("success_type") != success or row.get("id") in seen:
                continue
            segs = row.get("segments") or []
            prefix_parts = []
            target = None
            for seg in segs:
                kind = seg.get("kind")
                text = seg.get("text") or ""
                mask = seg.get("loss_mask") or []
                if kind == "action" and mask and sum(mask) == len(mask):
                    target = text
                    break
                prefix_parts.append(text)
            if not target:
                continue
            prefix = "".join(prefix_parts)
            if len(tokenizer.encode(prefix, add_special_tokens=False)) > max_prefix_tokens:
                continue
            seen.add(row.get("id"))
            picked.append({"id": row.get("id"), "success_type": success,
                           "db_id": row.get("db_id"), "prefix": prefix, "target": target})
            break
        if len(picked) >= n:
            break
    # Fill remaining slots from any success type.
    for row in rows:
        if len(picked) >= n:
            break
        if row.get("id") in seen:
            continue
        segs = row.get("segments") or []
        prefix_parts, target = [], None
        for seg in segs:
            if seg.get("kind") == "action" and seg.get("loss_mask") and \
                    sum(seg["loss_mask"]) == len(seg["loss_mask"]):
                target = seg.get("text") or ""
                break
            prefix_parts.append(seg.get("text") or "")
        if not target:
            continue
        prefix = "".join(prefix_parts)
        if len(tokenizer.encode(prefix, add_special_tokens=False)) > max_prefix_tokens:
            continue
        seen.add(row.get("id"))
        picked.append({"id": row.get("id"), "success_type": row.get("success_type"),
                       "db_id": row.get("db_id"), "prefix": prefix, "target": target})
    return picked


def summarize(items):
    n = len(items)
    if not n:
        return {"n": 0}
    out = {
        "n": n,
        "gen_one_complete_action": sum(x["gen"]["one_complete_action"] for x in items),
        "gen_first_action_valid": sum(x["gen"]["first_action_valid"] for x in items),
        "gen_raw_sql": sum(x["gen"]["first_body_format"] == "raw_sql" for x in items),
        "gen_json_wrapped": sum(x["gen"]["first_body_format"] == "json_wrapped" for x in items),
        "gen_no_complete_action": sum(not x["gen"]["has_first_action"] for x in items),
    }
    if items and "teacher" in items[0]:
        out.update({
            "teacher_loss_mean": sum(x["teacher"]["loss"] for x in items) / n,
            "teacher_token_acc_mean": sum(x["teacher"]["token_acc"] for x in items) / n,
            "teacher_first_token_acc_mean": sum(x["teacher"]["first_token_acc"] for x in items) / n,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/ma-user/work/env_verify/models/Qwen3-4B-Instruct-2507-ms")
    p.add_argument("--merged", default="outputs/qwen3_4b_sft/e1_lora_v1_resume105_merged")
    p.add_argument("--train-data", default="artifacts/m6/e1_alignment/e1_alignment_sft.jsonl")
    p.add_argument("--dev-traces", default="reports/m6/qwen3_4b_sft/resume105/smoke5_stop_traces/episodes.jsonl")
    p.add_argument("--out", default="reports/m6/qwen3_4b_sft/format_diagnostic.json")
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()

    import torch_npu  # noqa: F401
    torch.manual_seed(0)

    train_rows = read_jsonl(args.train_data)
    dev_rows = read_jsonl(args.dev_traces)
    dev_prompts = []
    for row in dev_rows:
        for seg in row.get("segments", []):
            if seg.get("kind") == "prompt":
                text = seg.get("text") or ""
                if text not in [x["prompt"] for x in dev_prompts]:
                    dev_prompts.append({"id": row.get("id"), "prompt": text})
                break
    dev_prompts = dev_prompts[:4]
    if not dev_prompts:
        raise SystemExit("no dev prompts found")
    print(json.dumps({"train_rows": len(train_rows), "dev_prompts": len(dev_prompts)}), flush=True)

    from transformers import AutoTokenizer
    pick_tok = AutoTokenizer.from_pretrained(args.base)
    examples = pick_training_examples(train_rows, pick_tok, args.n_train)
    print(json.dumps({"picked_examples": [
        {"id": x["id"], "success_type": x["success_type"], "db_id": x["db_id"]}
        for x in examples]}), flush=True)

    results = {}
    for arm, path in (("base", args.base), ("sft", args.merged)):
        print(json.dumps({"loading": arm, "path": path}), flush=True)
        model, tokenizer = load_model(path, args.device)
        arm_items = []
        for ex in examples:
            teacher = teacher_forced(model, tokenizer, ex["prefix"], ex["target"], args.device)
            gen = generate(model, tokenizer, ex["prefix"], args.device, args.max_new_tokens)
            arm_items.append({"id": ex["id"], "success_type": ex["success_type"],
                              "db_id": ex["db_id"], "teacher": teacher, "gen": gen})
            print(json.dumps({"arm": arm, "train_id": ex["id"], "tag": gen["first_tag"],
                              "teacher_acc": round(teacher["token_acc"], 3),
                              "gen_one": gen["one_complete_action"],
                              "gen_fmt": gen["first_body_format"]}), flush=True)
        dev_items = []
        for dp in dev_prompts:
            gen = generate(model, tokenizer, dp["prompt"], args.device, args.max_new_tokens)
            dev_items.append({"id": dp["id"], "gen": gen})
            print(json.dumps({"arm": arm, "dev_id": dp["id"], "tag": gen["first_tag"],
                              "gen_one": gen["one_complete_action"],
                              "gen_fmt": gen["first_body_format"]}), flush=True)
        results[arm] = {"train": summarize(arm_items), "dev": summarize(dev_items),
                        "train_items": arm_items, "dev_items": dev_items,
                        "hbm_peak_gb": float(torch_npu.npu.max_memory_allocated() / 1024 ** 3)}
        free_model(model)

    results["examples"] = [{"id": x["id"], "success_type": x["success_type"],
                            "db_id": x["db_id"]} for x in examples]
    results["meta"] = {
        "base": args.base, "merged": args.merged,
        "train_data": args.train_data, "dev_traces": args.dev_traces,
        "note": ("Training-prefix teacher forcing is a memorization check, not held-out "
                 "generalization. Generation uses greedy decoding; 'stop' is analyzed by "
                 "inspecting the first complete action, not by an inference stop string."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": args.out, "base": results["base"]["train"],
                      "sft": results["sft"]["train"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
