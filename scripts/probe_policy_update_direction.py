"""Measure whether a GRPO update moves action-token logprobs in the right direction.

For each recorded rollout in the frozen `sft_before/records.jsonl`:
  - reconstruct the exact model input from `segments` (prompt + observations),
  - mark action-token positions from `segments[kind == "action"].ids`,
  - compute the mean action-token logprob under the before and after models,
  - aggregate the delta by rollout correctness.

Expected direction: correct rollouts get a higher logprob (delta > 0), incorrect
rollouts get a lower logprob (delta < 0), and the correct-minus-incorrect gap is
positive. This is a diagnostic of the optimizer/advantage path, not a capability
metric.

The frozen `sft_before` records have no `loss_mask`/`response_tokens`; the action
segment ids are the supervision positions. The script fails loudly if a record
has no action tokens or if the token sequences cannot be reconstructed exactly.

`--mock` / `--mock-fail` run a CPU-only aggregation test without loading a model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_sequence(record):
    """Return (input_ids, action_mask) from the record's ordered segments."""
    segments = record.get("segments") or []
    if not segments:
        raise ValueError(f"record {record.get('id')} has no segments")
    input_ids, action_mask = [], []
    for index, segment in enumerate(segments):
        ids = segment.get("ids")
        kind = segment.get("kind")
        if not isinstance(ids, list) or not ids:
            raise ValueError(
                f"record {record.get('id')} segment {index} ({kind}) has no ids")
        if not all(isinstance(token, int) for token in ids):
            raise ValueError(f"record {record.get('id')} segment {index} has non-int ids")
        input_ids.extend(ids)
        action_mask.extend([kind == "action"] * len(ids))
    if not any(action_mask):
        raise ValueError(f"record {record.get('id')} has no action tokens")
    if len(input_ids) < 2:
        raise ValueError(f"record {record.get('id')} sequence is too short")
    return input_ids, action_mask


def load_model(path, device_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_name == "cpu":
        device = torch.device("cpu")
    else:
        try:
            import torch_npu  # noqa: F401
            if not torch.npu.is_available():
                raise RuntimeError("torch_npu is installed but no NPU is available")
            device = torch.device("npu:0")
        except ImportError:
            if device_name == "npu":
                raise
            device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", local_files_only=True,
        trust_remote_code=False).to(device).eval()
    return tokenizer, model, device


def action_logprobs(model, input_ids, action_mask, device):
    """Mean logprob of the supervised action tokens under one model.

    The model is run once on the full sequence; only action positions are
    scored. Logits are kept in the model dtype and each action position is
    converted to float32 before log_softmax to avoid a full float32 vocab
    matrix.
    """
    import torch

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    with torch.inference_mode():
        outputs = model(input_ids=ids, attention_mask=attention, use_cache=False)
        logits = outputs.logits[0]
    values = []
    for position, supervised in enumerate(action_mask):
        if not supervised:
            continue
        if position == 0:
            raise ValueError("an action token cannot be the first token")
        token = input_ids[position]
        row = logits[position - 1].float()
        value = torch.log_softmax(row, dim=-1)[token].item()
        if value != value or value == float("-inf"):
            raise ValueError(f"non-finite action logprob at position {position}")
        values.append(value)
    if not values:
        raise ValueError("no action logprobs computed")
    return sum(values) / len(values), len(values)


def score_records(records, model, tokenizer, device):
    """Score every record with one model; tokenizer is accepted for symmetry."""
    del tokenizer
    results = []
    for index, record in enumerate(records):
        input_ids, action_mask = build_sequence(record)
        mean_logprob, n_action = action_logprobs(model, input_ids, action_mask, device)
        results.append({
            "id": record.get("id"),
            "sample_index": record.get("sample_index"),
            "correct": bool(record.get("correct")),
            "n_action_tokens": n_action,
            "mean_action_logprob": mean_logprob,
        })
        if (index + 1) % 16 == 0 or index + 1 == len(records):
            print(json.dumps({"scored": index + 1, "of": len(records)}), flush=True)
    return results


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped["correct" if row["correct"] else "incorrect"].append(row["mean_action_logprob"])
    if not grouped["correct"] or not grouped["incorrect"]:
        raise ValueError("direction probe needs both correct and incorrect rollouts")
    mean_correct = sum(grouped["correct"]) / len(grouped["correct"])
    mean_incorrect = sum(grouped["incorrect"]) / len(grouped["incorrect"])
    return {
        "n": len(rows),
        "n_correct": len(grouped["correct"]),
        "n_incorrect": len(grouped["incorrect"]),
        "mean_before_logprob_correct": mean_correct,
        "mean_before_logprob_incorrect": mean_incorrect,
    }


def probe(records, before_model_path, after_model_path, device_name):
    before_tokenizer, before_model, device = load_model(before_model_path, device_name)
    before_rows = score_records(records, before_model, before_tokenizer, device)
    del before_model
    after_tokenizer, after_model, device_after = load_model(after_model_path, device_name)
    after_rows = score_records(records, after_model, after_tokenizer, device_after)
    del after_model
    if len(before_rows) != len(after_rows):
        raise ValueError("before/after record count mismatch")
    by_key = {}
    for row in before_rows:
        by_key[(row["id"], row["sample_index"])] = {"before": row}
    for row in after_rows:
        key = (row["id"], row["sample_index"])
        if key not in by_key:
            raise ValueError(f"unpaired after record: {key}")
        by_key[key]["after"] = row
    paired = []
    for key, pair in by_key.items():
        before, after = pair["before"], pair["after"]
        if before["n_action_tokens"] != after["n_action_tokens"]:
            raise ValueError(f"action token count changed for {key}")
        paired.append({
            "id": before["id"],
            "sample_index": before["sample_index"],
            "correct": before["correct"],
            "n_action_tokens": before["n_action_tokens"],
            "before_logprob": before["mean_action_logprob"],
            "after_logprob": after["mean_action_logprob"],
            "delta": after["mean_action_logprob"] - before["mean_action_logprob"],
        })
    correct = [r["delta"] for r in paired if r["correct"]]
    incorrect = [r["delta"] for r in paired if not r["correct"]]
    if not correct or not incorrect:
        raise ValueError("direction probe needs both correct and incorrect rollouts")
    delta_correct = sum(correct) / len(correct)
    delta_incorrect = sum(incorrect) / len(incorrect)
    gap = delta_correct - delta_incorrect
    per_question = defaultdict(list)
    for row in paired:
        per_question[row["id"]].append(row)
    per_question_summary = {}
    for question_id, rows in sorted(per_question.items()):
        correct_rows = [r["delta"] for r in rows if r["correct"]]
        incorrect_rows = [r["delta"] for r in rows if not r["correct"]]
        per_question_summary[question_id] = {
            "n": len(rows),
            "n_correct": len(correct_rows),
            "n_incorrect": len(incorrect_rows),
            "delta_correct": sum(correct_rows) / len(correct_rows) if correct_rows else None,
            "delta_incorrect": sum(incorrect_rows) / len(incorrect_rows) if incorrect_rows else None,
        }
    return {
        "n": len(paired),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "delta_correct": delta_correct,
        "delta_incorrect": delta_incorrect,
        "gap_delta": gap,
        "pass_direction": bool(delta_correct > 0 and delta_incorrect < 0 and gap > 0),
        "per_question": per_question_summary,
        "records": paired,
    }


def mock_probe(fail=False):
    """CPU-only aggregation test with synthetic before/after logprobs."""
    rows = []
    for index in range(10):
        correct = index < 6
        before = -1.0 - 0.01 * index
        delta = (0.1 if correct else -0.1) * (-1 if fail else 1)
        rows.append({
            "id": f"q{index % 4}",
            "sample_index": index,
            "correct": correct,
            "n_action_tokens": 10,
            "before_logprob": before,
            "after_logprob": before + delta,
            "delta": delta,
        })
    correct = [r["delta"] for r in rows if r["correct"]]
    incorrect = [r["delta"] for r in rows if not r["correct"]]
    delta_correct = sum(correct) / len(correct)
    delta_incorrect = sum(incorrect) / len(incorrect)
    gap = delta_correct - delta_incorrect
    return {
        "mock": True,
        "n": len(rows),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "delta_correct": delta_correct,
        "delta_incorrect": delta_incorrect,
        "gap_delta": gap,
        "pass_direction": bool(delta_correct > 0 and delta_incorrect < 0 and gap > 0),
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-records", default=(
        "reports/m6/rl_learning20_4q_20261118/sft_before/records.jsonl"))
    parser.add_argument("--before-model", default=(
        "outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"))
    parser.add_argument("--after-model", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-fail", action="store_true")
    parser.add_argument("--require-direction", action="store_true")
    args = parser.parse_args()

    if args.mock or args.mock_fail:
        result = mock_probe(fail=args.mock_fail)
    else:
        if not args.after_model:
            parser.error("--after-model is required unless --mock is used")
        records = read_jsonl(args.before_records)
        if args.limit:
            records = records[: args.limit]
        result = probe(records, args.before_model, args.after_model, args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {key: result[key] for key in (
        "n", "n_correct", "n_incorrect", "delta_correct", "delta_incorrect",
        "gap_delta", "pass_direction")}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.require_direction and not result["pass_direction"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
