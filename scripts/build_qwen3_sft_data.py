"""L0: re-tokenize the tool-consistent E1 SFT text for Qwen3-4B (no text edits)."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    model_root = Path(os.environ.get("SQLRL_MODEL_ROOT", Path.home() / "work" / "env_verify" / "models"))
    p.add_argument("--data", default="artifacts/m6/e1_alignment/e1_alignment_sft.jsonl")
    p.add_argument("--tokenizer", default=str(model_root / "Qwen3-4B-Instruct-2507-ms"))
    p.add_argument("--out", default="artifacts/m6/qwen3_4b_sft/e1_sft_qwen3.jsonl")
    p.add_argument("--manifest", default="artifacts/m6/qwen3_4b_sft/dataset_manifest.json")
    p.add_argument("--max-context", type=int, default=6144)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = read_jsonl(a.data)
    out_rows, excluded = [], []
    token_stats = Counter()
    mask_stats = Counter()
    source_counts = Counter()
    tool_shapes = Counter()
    decode_mismatch = 0
    for row in rows:
        segs = row.get("segments") or []
        ids, masks = [], []
        text_parts = []
        for seg in segs:
            text = seg.get("text") or ""
            seg_mask = seg.get("loss_mask") or []
            if seg_mask and sum(seg_mask) not in (0, len(seg_mask)):
                raise SystemExit(f"partial segment mask unsupported: {row.get('id')}")
            supervised = bool(seg_mask) and sum(seg_mask) == len(seg_mask)
            seg_ids = tok.encode(text, add_special_tokens=False)
            if tok.decode(seg_ids, skip_special_tokens=True) != text:
                decode_mismatch += 1
            ids.extend(seg_ids)
            masks.extend([1 if supervised else 0] * len(seg_ids))
            text_parts.append(text)
        if not ids or len(ids) != len(masks):
            excluded.append({"id": row.get("id"), "reason": "empty_or_mask_mismatch"})
            continue
        if len(ids) > a.max_context:
            excluded.append({"id": row.get("id"), "reason": "over_budget", "tokens": len(ids)})
            continue
        record = {
            "id": row.get("id"),
            "question_id": row.get("question_id"),
            "db_id": row.get("db_id"),
            "trajectory_source": row.get("trajectory_source"),
            "runtime_fingerprint_sha256": row.get("runtime_fingerprint_sha256"),
            "source_text_sha256": sha_text("".join(text_parts)),
            "input_ids": ids,
            "loss_mask": masks,
            "n_segments": len(segs),
            "input_tokens": len(ids),
            "supervised_tokens": sum(masks),
        }
        out_rows.append(record)
        source_counts[row.get("trajectory_source")] += 1
        actions = [e.get("action") for e in (row.get("tool_events") or [])]
        has_l, has_x = "schema_lookup" in actions, "sql_execute" in actions
        tool_shapes["lookup_and_execute" if has_l and has_x else "execute_only" if has_x
                   else "lookup_only" if has_l else "zero_tool"] += 1
        token_stats["input_tokens"] += len(ids)
        token_stats["supervised_tokens"] += sum(masks)
        mask_stats["supervised_segments"] += sum(1 for seg in segs
                                                 if seg.get("loss_mask") and sum(seg["loss_mask"]) == len(seg["loss_mask"]))
        mask_stats["masked_segments"] += sum(1 for seg in segs
                                             if not seg.get("loss_mask") or sum(seg["loss_mask"]) == 0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    source_sha = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    manifest = {
        "stage": "M6-08l-L0",
        "source": a.data,
        "source_sha256": source_sha,
        "tokenizer": a.tokenizer,
        "records": len(out_rows),
        "excluded": excluded,
        "token_stats": dict(token_stats),
        "avg_input_tokens": token_stats["input_tokens"] / max(len(out_rows), 1),
        "avg_supervised_tokens": token_stats["supervised_tokens"] / max(len(out_rows), 1),
        "supervised_token_ratio": token_stats["supervised_tokens"] / max(token_stats["input_tokens"], 1),
        "mask_segments": dict(mask_stats),
        "source_counts": dict(source_counts),
        "by_tool_shape": dict(tool_shapes),
        "decode_mismatch_segments": decode_mismatch,
        "note": "Text is copied verbatim from E1 segments; only tokenizer changed. "
                "Tool/action consistency is inherited from the E1 audit.",
    }
    Path(a.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("records", "token_stats", "avg_input_tokens",
                                              "avg_supervised_tokens", "supervised_token_ratio",
                                              "source_counts", "by_tool_shape",
                                              "decode_mismatch_segments")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
