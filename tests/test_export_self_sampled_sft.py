"""CPU tests for the self-sampled SFT export (no GPU, no model)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts" / "export_self_sampled_sft.py"


def write_fixture(tmp_path, n_questions=2, samples=2):
    records = []
    for question_index in range(n_questions):
        for sample in range(samples):
            correct = question_index == 0 and sample == 0
            records.append({
                "id": f"q{question_index}", "db_id": "db",
                "question": "question", "gold_sql": "SELECT 1",
                "final_sql": "SELECT 1" if correct else "SELECT 2",
                "segments": [{"kind": "prompt", "text": "p"}],
                "tool_events": [], "executions": [],
                "sample_index": sample, "correct": correct, "valid": True,
                "reward_available": True,
                "score": {"correct": correct, "ok": True, "error_type": None,
                          "gold_rows": 1},
            })
    (tmp_path / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "n_questions": n_questions,
        "runtime_contract": "A-e1.2-typed-v1",
        "runtime_fingerprint": "fingerprint",
    }))
    (tmp_path / "out").mkdir()


def run_export(tmp_path, samples):
    return subprocess.run(
        [sys.executable, str(EXPORT),
         "--records", str(tmp_path / "records.jsonl"),
         "--manifest", str(tmp_path / "manifest.json"),
         "--out", str(tmp_path / "out"),
         "--min-samples", str(samples), "--max-per-question", "1",
         "--split-name", "hard_reviewed_11_seen"],
        capture_output=True, text=True)


def has_forbidden(value):
    forbidden = {"gold_sql", "gold_rows", "gold_error"}
    if isinstance(value, dict):
        return bool(forbidden & set(value)) or any(has_forbidden(v) for v in value.values())
    if isinstance(value, list):
        return any(has_forbidden(v) for v in value)
    return False


def test_export_selects_correct_and_strips_gold(tmp_path):
    write_fixture(tmp_path)
    result = run_export(tmp_path, samples=2)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "sft_correct.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "q0"
    assert rows[0]["training_eligible"] is True
    assert rows[0]["sft_split"] == "hard_reviewed_11_seen"
    assert "score" not in rows[0]
    assert not has_forbidden(rows[0])
    summary = json.loads((tmp_path / "out" / "yield_summary.json").read_text())
    assert summary["questions"] == 2
    assert summary["correct_trajectories_total"] == 1
    assert summary["all_zero_questions"] == 1


def test_export_rejects_incomplete_samples(tmp_path):
    write_fixture(tmp_path)
    result = run_export(tmp_path, samples=3)
    assert result.returncode != 0
    assert "expected 3" in result.stderr


def test_export_rejects_nested_gold_field(tmp_path):
    write_fixture(tmp_path)
    rows = [json.loads(line) for line in
            (tmp_path / "records.jsonl").read_text().splitlines() if line.strip()]
    rows[0]["tool_events"] = [{"result": {"gold_sql": "SELECT 1"}}]
    (tmp_path / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_export(tmp_path, samples=2)
    assert result.returncode != 0
    assert "Gold-derived" in result.stderr
