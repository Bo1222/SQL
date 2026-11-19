import json
from pathlib import Path

import pytest
import torch

from e1_alignment.provenance import RAW_SUCCESS, build_provenance
from e1_alignment.rebuild.training_math import backward_window, supervised_tokens
from e1_alignment.student_sft_data import (
    SFTDataError,
    StudentDataset,
    extract_ids_and_mask,
    load_examples,
    validate_example,
)


def provenance():
    return build_provenance(
        source=RAW_SUCCESS,
        generation_mode="n1_greedy",
        model_path="m",
        model_fingerprint="f" * 64,
        runtime_fingerprint="r" * 64,
        retrieval_version="E1.1",
        prompt_mode="query",
        observation_mode="compact",
        split="sft_train",
        final_ex=True,
        final_ex_source="gold_scoring",
    )


def new_record():
    return {
        "id": "q1",
        "final_ex": True,
        "final_sql": "SELECT 1",
        "provenance": provenance(),
        "trajectory": {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "loss_mask": [0, 0, 1, 1, 0, 1],
            "segments": [
                {"kind": "prompt", "ids": [1, 2], "text": "p"},
                {"kind": "action", "ids": [3, 4], "text": "a1"},
                {"kind": "observation", "ids": [5], "text": "o"},
                {"kind": "action", "ids": [6], "text": "a2"},
            ],
        },
    }


def segment_record():
    return {
        "id": "old",
        "final_ex": True,
        "trajectory": {
            "segments": [
                {"kind": "prompt", "ids": [1, 2], "text": "p", "supervise": False},
                {"kind": "action", "ids": [3], "text": "failed", "supervise": False},
                {"kind": "observation", "ids": [4], "text": "o", "supervise": False},
                {"kind": "action", "ids": [5, 6], "text": "ok", "supervise": True},
            ]
        },
    }


def test_stored_loss_mask_is_used():
    ids, mask, source = extract_ids_and_mask(new_record())
    assert source == "trajectory.loss_mask"
    assert ids == [1, 2, 3, 4, 5, 6]
    assert mask == [0, 0, 1, 1, 0, 1]


def test_segment_fallback_never_supervises_all_actions():
    ids, mask, source = extract_ids_and_mask(segment_record())
    assert source == "segment.supervise"
    assert mask == [0, 0, 0, 0, 1, 1]


def test_validate_example_preserves_mask():
    row = validate_example(new_record(), max_context=10)
    assert row["mask_source"] == "trajectory.loss_mask"
    assert row["supervised_tokens"] == 3


def test_rejects_prompt_supervision():
    record = new_record()
    record["trajectory"]["loss_mask"][0] = 1
    with pytest.raises(SFTDataError):
        validate_example(record, max_context=10)


def test_rejects_all_zero_mask():
    record = new_record()
    record["trajectory"]["loss_mask"] = [0] * 6
    with pytest.raises(SFTDataError):
        validate_example(record, max_context=10)


def test_rejects_length_mismatch():
    record = new_record()
    record["trajectory"]["loss_mask"] = [0, 0, 1]
    with pytest.raises(SFTDataError):
        validate_example(record, max_context=10)


def test_rejects_mask_disagreeing_with_segment_supervision():
    record = segment_record()
    record["trajectory"]["segments"][1]["supervise"] = True
    # The fallback mask follows segment supervision, so this is consistent;
    # construct a stored-mask record that contradicts a known supervise flag.
    record["trajectory"]["input_ids"] = [1, 2, 3, 4, 5, 6]
    record["trajectory"]["loss_mask"] = [0, 0, 0, 0, 1, 1]
    with pytest.raises(SFTDataError):
        validate_example(record, max_context=10)


def test_rejects_gold_when_trainable_required():
    record = new_record()
    record["provenance"]["source"] = "gold_constructed"
    record["provenance"]["trainable"] = False
    record["provenance"]["final_ex_source"] = "inherited_legacy"
    record["provenance"]["generation_mode"] = "legacy_replay"
    record["provenance"]["split"] = None
    with pytest.raises(Exception):
        validate_example(record, max_context=10, require_trainable=True)


def test_dataset_labels_hide_masked_tokens():
    row = validate_example(new_record(), max_context=10)
    dataset = StudentDataset([row])
    item = dataset[0]
    assert item["labels"].tolist() == [-100, -100, 3, 4, -100, 6]


def test_backward_window_is_token_weighted():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(2.0))

    def loss_fn(model, batch):
        return model.w * batch["x"]

    batches = [
        {"labels": torch.tensor([[0, 1, 1]]), "x": torch.tensor(3.0)},
        {"labels": torch.tensor([[0, -100, 1]]), "x": torch.tensor(5.0)},
    ]
    counts = [supervised_tokens(b) for b in batches]
    assert counts == [2, 1]
    model = Tiny()
    loss, total = backward_window(model, batches, loss_fn)
    weighted_x = (2 / 3) * 3.0 + (1 / 3) * 5.0
    assert total == 3
    assert abs(loss - 2.0 * weighted_x) < 1e-6
    assert abs(float(model.w.grad) - weighted_x) < 1e-6


def test_load_examples_reports_exclusions(tmp_path):
    good = new_record()
    bad = new_record()
    bad["id"] = "bad"
    bad["trajectory"]["loss_mask"] = [0] * 6
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    rows, excluded = load_examples(path, max_context=10)
    assert len(rows) == 1
    assert len(excluded) == 1
    assert excluded[0]["id"] == "bad"


def test_backward_window_loss_scale():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(2.0))

    def loss_fn(model, batch):
        return model.w * batch["x"]

    batches = [
        {"labels": torch.tensor([[0, 1, 1]]), "x": torch.tensor(3.0)},
        {"labels": torch.tensor([[0, -100, 1]]), "x": torch.tensor(5.0)},
    ]
    model = Tiny()
    loss, total = backward_window(model, batches, loss_fn, loss_scale=2.0)
    weighted_x = (2 / 3) * 3.0 + (1 / 3) * 5.0
    assert total == 3
    assert abs(float(model.w.grad) - 2.0 * weighted_x) < 1e-6
    assert abs(loss - 2.0 * 2.0 * weighted_x) < 1e-6
