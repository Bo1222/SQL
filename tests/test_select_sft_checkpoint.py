"""Checkpoint-selection statistics are paired and use dev200 only."""
from scripts.select_sft_checkpoint import stats


def record(qid, db, correct, valid=True, looked_up=True, tables=None):
    return {
        "id": qid,
        "db_id": db,
        "correct": correct,
        "valid": valid,
        "finish_reason": "answer_submitted",
        "tool_calls": 2,
        "error_type": None if correct else "no_such_column",
        "usage": {"looked_up": looked_up,
                  "tables": ["t"] if tables is None else tables},
    }


def base_records():
    rows = [record("a", "db1", True), record("b", "db1", False),
            record("c", "db2", True), record("d", "db2", False)]
    return {row["id"]: row for row in rows}


def test_identical_epoch_has_zero_delta_and_p_one():
    base = base_records()
    result = stats(dict(base), base)
    assert abs(result["ex"] - 0.5) < 1e-9
    assert result["gain"] == 0 and result["loss"] == 0
    assert result["net_gain"] == 0 and result["delta_ex"] == 0
    assert result["mcnemar_exact_p"] == 1.0


def test_gain_and_loss_are_paired_by_id():
    base = base_records()
    epoch = {key: dict(value) for key, value in base.items()}
    epoch["b"]["correct"] = True   # gain
    epoch["c"]["correct"] = False  # loss
    result = stats(epoch, base)
    assert result["gain"] == 1 and result["loss"] == 1
    assert result["net_gain"] == 0
    assert result["delta_ex"] == 0
    assert result["base_correct"] == 2 and result["correct"] == 2


def test_retrieval_and_validity_stats():
    base = base_records()
    epoch = {key: dict(value) for key, value in base.items()}
    epoch["a"]["valid"] = False
    epoch["a"]["usage"] = {"looked_up": True, "tables": []}
    result = stats(epoch, base)
    assert result["valid"] == 0.75
    assert result["empty_lookup_rate"] == 0.25
    assert result["lookup_rate"] == 1.0
