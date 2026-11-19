"""Sharding must be a disjoint, complete partition of the eval rows."""
import pytest

from eval.evaluate_agent import shard_rows


def test_shards_partition_rows_without_overlap():
    rows = list(range(203))
    parts = [shard_rows(rows, i, 4) for i in range(4)]
    assert sum(len(part) for part in parts) == len(rows)
    assert sorted(item for part in parts for item in part) == rows
    assert len({item for part in parts for item in part}) == len(rows)


def test_single_shard_returns_all_rows():
    rows = list(range(10))
    assert shard_rows(rows, 0, 1) == rows


def test_invalid_shard_rejected():
    with pytest.raises(ValueError):
        shard_rows([1, 2], 4, 4)
    with pytest.raises(ValueError):
        shard_rows([1, 2], 0, 0)
