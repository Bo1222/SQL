"""Mask-correct SFT data loading for M6-08m student-only trajectories.

The old trainer rebuilt ``loss_mask`` from ``segment.kind == "action"``, which
supervised failed actions as well.  This module always consumes the stored
token-level ``loss_mask`` when present and only falls back to
``segment["supervise"]`` for legacy records.  It never treats every action as
supervised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from e1_alignment.provenance import ProvenanceError, ensure_trainable, read_jsonl


class SFTDataError(ValueError):
    """Raised when a record violates the SFT data contract."""


def _trajectory(record: dict) -> dict:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
        raise SFTDataError("missing trajectory object")
    return trajectory


def extract_ids_and_mask(record: dict) -> tuple[list[int], list[int], str]:
    """Return ``(input_ids, loss_mask, mask_source)``.

    Priority:
    1. stored token-level ``trajectory.loss_mask`` (new format);
    2. ``segment.supervise`` for legacy records (never ``kind == action``).
    """
    trajectory = _trajectory(record)
    ids = list(trajectory.get("input_ids") or [])
    mask = trajectory.get("loss_mask")
    if ids and mask is not None:
        mask = list(mask)
        return ids, mask, "trajectory.loss_mask"

    segments = trajectory.get("segments") or []
    if not segments:
        raise SFTDataError("neither trajectory.loss_mask nor segments present")
    built_ids: list[int] = []
    built_mask: list[int] = []
    for segment in segments:
        segment_ids = list(segment.get("ids") or [])
        kind = segment.get("kind")
        supervise = segment.get("supervise")
        if supervise is None and kind == "action":
            # New rebuilt records carry segment-level action/supervise only in
            # the top-level mask; without either, refuse to guess.
            raise SFTDataError("action segment has no supervise flag")
        value = 1 if (kind == "action" and supervise is True) else 0
        built_ids.extend(segment_ids)
        built_mask.extend([value] * len(segment_ids))
    return built_ids, built_mask, "segment.supervise"


def validate_example(record: dict, *, max_context: int, require_trainable: bool = False) -> dict:
    """Validate one record and return the normalised training row."""
    if require_trainable:
        ensure_trainable(record)
    ids, mask, mask_source = extract_ids_and_mask(record)
    if not ids:
        raise SFTDataError("empty input_ids")
    if len(ids) != len(mask):
        raise SFTDataError(f"input_ids/mask length mismatch: {len(ids)} vs {len(mask)}")
    if any(value not in (0, 1) for value in mask):
        raise SFTDataError("loss_mask must contain only 0/1")
    if mask[0] != 0:
        raise SFTDataError("prompt token must not be supervised")
    if not any(mask):
        raise SFTDataError("no supervised tokens")
    if len(ids) > max_context:
        raise SFTDataError(f"over context budget: {len(ids)} > {max_context}")

    trajectory = _trajectory(record)
    segments = trajectory.get("segments") or []
    if segments:
        segment_tokens = sum(len(segment.get("ids") or []) for segment in segments)
        if segment_tokens != len(ids):
            raise SFTDataError(
                f"segment/input_ids length mismatch: segments={segment_tokens} ids={len(ids)}")
        expected = []
        for segment in segments:
            length = len(segment.get("ids") or [])
            kind = segment.get("kind")
            supervise = segment.get("supervise")
            if kind == "action" and supervise is None:
                expected.extend([None] * length)  # unknown, skip exact check
            else:
                expected.extend([1 if (kind == "action" and supervise is True) else 0] * length)
        if len(expected) == len(mask):
            for index, value in enumerate(expected):
                if value is not None and value != mask[index]:
                    raise SFTDataError(
                        f"mask disagrees with segment supervision at token {index}: "
                        f"stored={mask[index]} expected={value}"
                    )

    record_id = record.get("id") or record.get("trajectory_id") or record.get("question_id")
    provenance = record.get("provenance")
    return {
        "id": record_id,
        "input_ids": ids,
        "loss_mask": mask,
        "mask_source": mask_source,
        "supervised_tokens": sum(mask),
        "provenance_source": (provenance or {}).get("source"),
        "final_ex": record.get("final_ex"),
        "final_sql": record.get("final_sql"),
    }


def load_examples(path, *, max_context: int = 6144,
                  require_trainable: bool = False) -> tuple[list[dict], list[dict]]:
    rows, excluded = [], []
    for index, record in enumerate(read_jsonl(path)):
        try:
            rows.append(validate_example(record, max_context=max_context,
                                         require_trainable=require_trainable))
        except (SFTDataError, ProvenanceError) as exc:
            record_id = record.get("id") or record.get("trajectory_id") or record.get("question_id")
            excluded.append({"index": index, "id": record_id, "reason": str(exc)})
    return rows, excluded


@dataclass
class StudentDataset:
    rows: list[dict]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["loss_mask"], dtype=torch.bool)
        labels = torch.where(mask, ids, torch.full_like(ids, -100))
        return {"input_ids": ids, "labels": labels, "id": row["id"]}


def collate(batch):
    assert len(batch) == 1, "micro batch size is fixed at 1 for the NPU budget"
    item = batch[0]
    return {"input_ids": item["input_ids"][None, :],
            "attention_mask": __import__("torch").ones_like(item["input_ids"])[None, :],
            "labels": item["labels"][None, :], "id": item["id"]}


def masked_loss(model, batch):
    import torch

    device = next(model.parameters()).device
    output = model(input_ids=batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device),
                   labels=batch["labels"].to(device), use_cache=False)
    return output.loss


def summary(rows: list[dict], excluded: list[dict]) -> dict:
    from collections import Counter

    sources = Counter(row.get("provenance_source") for row in rows)
    return {
        "total_loaded": len(rows),
        "excluded": len(excluded),
        "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        "provenance_counts": dict(sources),
        "mask_sources": dict(Counter(row["mask_source"] for row in rows)),
        "excluded_reasons": dict(Counter(item["reason"].split(":")[0] for item in excluded)),
    }


def write_jsonl(path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
