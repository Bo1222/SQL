"""CPU tests for the schema-linking SFT builder selection helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_schema_linking_sft import (  # noqa: E402
    lookup_has_tables,
    select_pairs,
)


def test_lookup_has_tables_reads_metadata_or_tables():
    assert lookup_has_tables({"status": "ok",
                              "retrieval_metadata": {"matched_tables": ["t"]}})
    assert lookup_has_tables({"status": "ok", "tables": [{"name": "t"}]})
    assert not lookup_has_tables({"status": "ok", "tables": []})
    assert not lookup_has_tables({"status": "ok"})


def test_select_pairs_stratifies_and_caps_per_db():
    candidates = [
        {"question_id": f"q{index}", "db_id": "a" if index < 4 else "b"}
        for index in range(8)
    ]
    selected = select_pairs(candidates, limit=3, max_per_db=2, seed=0)
    assert len(selected) == 3
    assert sum(row["db_id"] == "a" for row in selected) == 2
    assert sum(row["db_id"] == "b" for row in selected) == 1
    # Selection must be deterministic for a fixed seed.
    again = select_pairs(candidates, limit=3, max_per_db=2, seed=0)
    assert [row["question_id"] for row in selected] == [
        row["question_id"] for row in again]
