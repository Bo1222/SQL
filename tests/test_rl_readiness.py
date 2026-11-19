import json

import pandas as pd
import pytest

from scripts.check_rl_readiness import check_disjoint, check_parquet, check_sft_gate


def _frame(split, ids, dbs, *, prompt_mode="query", retrieval="E1.1",
           prompt_version="catalog-three-action-query-v2"):
    rows = []
    for qid, db in zip(ids, dbs):
        rows.append({
            "data_source": "bird_sql",
            "prompt": [{"role": "user", "content": "x" * 10}],
            "ability": "text2sql",
            "reward_model": {"style": "rule", "ground_truth": "SELECT 1"},
            "extra_info": {
                "question_id": qid,
                "db_id": db,
                "db_path": __file__,
                "split": split,
                "prompt_mode": prompt_mode,
                "retrieval_version": retrieval,
                "prompt_version": prompt_version,
            },
        })
    return pd.DataFrame(rows)


def test_check_parquet_accepts_valid_contract(tmp_path):
    path = tmp_path / "rl.parquet"
    _frame("rl", ["q1", "q2"], ["db1", "db1"]).to_parquet(path, index=False)
    errors, warnings = [], []
    info = check_parquet(path, expected_split="rl", name="train", errors=errors, warnings=warnings)
    assert info["rows"] == 2
    assert info["db_count"] == 1
    assert errors == []


def test_check_parquet_rejects_wrong_prompt_contract(tmp_path):
    path = tmp_path / "rl.parquet"
    _frame("rl", ["q1"], ["db1"], prompt_mode="sft", retrieval="E1").to_parquet(path, index=False)
    errors, warnings = [], []
    check_parquet(path, expected_split="rl", name="train", errors=errors, warnings=warnings)
    assert any("prompt_mode" in error for error in errors)
    assert any("retrieval_version" in error for error in errors)


def test_check_parquet_validates_typed_prompt_and_fingerprint(tmp_path, monkeypatch):
    from agent.runtime_contract import CONTRACT_ID, FINGERPRINT, prompt
    from types import SimpleNamespace
    store = SimpleNamespace(table_names=["items"])
    monkeypatch.setattr("agent.runtime_contract.load_store", lambda *args: store)
    frame = _frame("rl", ["q1"], ["db1"], retrieval="E1.2")
    info = frame.iloc[0]["extra_info"]
    info.update(runtime_contract=CONTRACT_ID, runtime_fingerprint=FINGERPRINT,
                score_mode="bird_set", question="How many items?", evidence="")
    frame.iloc[0]["prompt"][0]["content"] = prompt(info, store)
    path = tmp_path / "typed.parquet"
    frame.to_parquet(path, index=False)
    errors = []
    check_parquet(path, expected_split="rl", name="train", errors=errors, warnings=[], runtime_contract=CONTRACT_ID)
    assert not errors
    frame.iloc[0]["prompt"][0]["content"] += " extra schema"
    frame.to_parquet(path, index=False)
    check_parquet(path, expected_split="rl", name="train", errors=errors, warnings=[], runtime_contract=CONTRACT_ID)
    assert any("prompt differs" in error for error in errors)
    info["runtime_fingerprint"] = "bad"
    frame.to_parquet(path, index=False)
    errors = []
    check_parquet(path, expected_split="rl", name="train", errors=errors, warnings=[], runtime_contract=CONTRACT_ID)
    assert any("fingerprint" in error for error in errors)


def test_check_disjoint_reports_db_and_question_overlap():
    train = {"dbs": ["a", "b"], "ids": ["q1", "q2"]}
    val = {"dbs": ["b", "c"], "ids": ["q2", "q3"]}
    errors = []
    check_disjoint(train, val, "dev200", errors)
    assert any("DB overlap" in error for error in errors)
    assert any("question overlap" in error for error in errors)


def test_check_sft_gate_requires_positive_delta(tmp_path):
    base = tmp_path / "base.json"
    sft = tmp_path / "sft.json"
    base.write_text(json.dumps({"n": 1534, "ex": 0.40}))
    sft.write_text(json.dumps({"n": 1534, "ex": 0.39}))
    errors = []
    info = check_sft_gate(base, sft, require=True, errors=errors, warnings=[])
    assert info["delta"] == pytest.approx(-0.01)
    assert any("not positive" in error for error in errors)


def test_check_sft_gate_passes_positive_delta(tmp_path):
    base = tmp_path / "base.json"
    sft = tmp_path / "sft.json"
    base.write_text(json.dumps({"n": 1534, "ex": 0.30}))
    sft.write_text(json.dumps({"n": 1534, "ex": 0.32}))
    errors = []
    info = check_sft_gate(base, sft, require=True, errors=errors, warnings=[])
    assert info["delta"] == pytest.approx(0.02)
    assert errors == []


def test_check_sft_gate_uses_paired_p_value_and_warns_on_ci(tmp_path):
    base = tmp_path / "base.json"
    sft = tmp_path / "sft.json"
    paired = tmp_path / "paired.json"
    base.write_text(json.dumps({"n": 1534, "ex": 0.30}))
    sft.write_text(json.dumps({"n": 1534, "ex": 0.32}))
    paired.write_text(json.dumps({
        "mcnemar_exact_p": 0.0675, "net_gain": 30, "b_only": 141, "a_only": 111,
        "db_bootstrap_ci95": [-0.028, 0.066],
    }))
    errors, warnings = [], []
    info = check_sft_gate(base, sft, require=True, errors=errors, warnings=warnings,
                          paired_path=paired)
    assert info["mcnemar_exact_p"] == pytest.approx(0.0675)
    assert errors == []
    assert any("includes 0" in warning for warning in warnings)


def test_check_sft_gate_rejects_insignificant_paired_p(tmp_path):
    base = tmp_path / "base.json"
    sft = tmp_path / "sft.json"
    paired = tmp_path / "paired.json"
    base.write_text(json.dumps({"n": 1534, "ex": 0.30}))
    sft.write_text(json.dumps({"n": 1534, "ex": 0.32}))
    paired.write_text(json.dumps({
        "mcnemar_exact_p": 0.2, "net_gain": 10, "b_only": 20, "a_only": 10,
        "db_bootstrap_ci95": [-0.02, 0.06],
    }))
    errors = []
    check_sft_gate(base, sft, require=True, errors=errors, warnings=[],
                   paired_path=paired)
    assert any("McNemar" in error for error in errors)


def test_check_local_dynamic_sampling_require(tmp_path):
    from scripts.check_rl_readiness import check_local_dynamic_sampling

    (tmp_path / "e1_alignment").mkdir()
    (tmp_path / "e1_alignment" / "rl_core.py").write_text("def select_mixed_groups(): pass\n")
    errors, warnings = [], []
    info = check_local_dynamic_sampling(tmp_path, require=False, errors=errors, warnings=warnings)
    assert info["missing"]
    assert warnings and not errors

    errors, warnings = [], []
    info = check_local_dynamic_sampling(tmp_path, require=True, errors=errors, warnings=warnings)
    assert info["missing"]
    assert errors and not warnings
