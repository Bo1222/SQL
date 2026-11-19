"""Dataset contract and split-leakage tests (plan §4.5/§4.7 and M1 acceptance)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rl_parquet_contract():
    path = Path("artifacts/datasets/m0_smoke.parquet")
    if not path.is_file():
        pytest.skip("smoke parquet not built")
    df = pd.read_parquet(path)
    for column in ("data_source", "prompt", "reward_model", "extra_info"):
        assert column in df.columns
    row = df.iloc[0]
    assert row["reward_model"]["style"] == "rule"
    assert row["reward_model"]["ground_truth"]
    # the prompt must not leak the gold SQL or any DDL
    text = row["prompt"][0]["content"]
    assert row["reward_model"]["ground_truth"] not in text
    assert "CREATE TABLE" not in text.upper()
    assert row["extra_info"]["db_id"] and row["extra_info"]["db_path"]


def test_split_audit_has_no_leakage():
    sys.path.insert(0, str(Path("data").resolve()))
    from data.validate_splits import summarize
    splits = {
        "sft": summarize(read_jsonl("data/processed/bird_agent/full_sft_20261026/full_tool.jsonl")),
        "rl": summarize(read_jsonl("data/processed/bird_agent/rl.jsonl")),
        "val": summarize(read_jsonl("data/processed/bird_agent/val.jsonl")),
        "test": summarize(read_jsonl("data/processed/bird_agent/test.jsonl")),
    }
    assert not (splits["sft"]["dbs"] & splits["val"]["dbs"])
    assert not (splits["sft"]["dbs"] & splits["test"]["dbs"])
    assert not (splits["rl"]["dbs"] & splits["val"]["dbs"])
    assert not (splits["rl"]["dbs"] & splits["test"]["dbs"])
    assert not (splits["val"]["dbs"] & splits["test"]["dbs"])
    # SFT covers every RL database (documented limitation, not a silent assumption)
    assert splits["sft"]["dbs"] == splits["rl"]["dbs"] | (splits["sft"]["dbs"] - splits["rl"]["dbs"])


def test_catalog_stores_match_splits():
    catalog = Path("artifacts/catalog")
    if not catalog.is_dir():
        pytest.skip("catalog not built")
    ids = {p.name for p in catalog.iterdir() if p.is_dir()}
    for split in ("rl", "val", "test"):
        dbs = {row["db_id"] for row in read_jsonl(f"data/processed/bird_agent/{split}.jsonl")}
        assert dbs <= ids, f"missing stores for {split}: {dbs - ids}"


def test_gold_sql_is_not_in_the_prompt_builder():
    from agent.protocol import build_prompt
    prompt = build_prompt("question", "evidence", "table_a\ntable_b")
    assert "### Table catalog" in prompt
    assert "CREATE TABLE" not in prompt.upper()
    assert "table_a" in prompt and "table_b" in prompt

