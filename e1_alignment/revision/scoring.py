"""Exact-string execution cache bound to the already-verified database hashes.

No whitespace/case/literal rewriting; failed transport results are never cached.
An in-process future coalesces duplicate concurrent scoring calls.
"""
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from e1_alignment.common import canonical_hash, read_jsonl, append_jsonl
from reward.bird_ex_reward import score_sql as execute_score

_lock = threading.Lock()
_futures = {}
_hashes = None
_loaded = False
CACHE = Path("artifacts/m6/sft_data_revision/exact_scores.jsonl")


def score_sql(db_path, candidate_sql, gold_sql, timeout=30):
    global _hashes, _loaded
    with _lock:
        if _hashes is None:
            fingerprint = json.loads(Path("reports/m6/e1_alignment/runtime_fingerprint.json").read_text())
            _hashes = {str(Path(r["path"]).resolve()): r["sha256"] for r in fingerprint["database"]["databases"]}
        resolved = str(Path(db_path).resolve())
        # Fixture/non-contract DBs have no cache authority and execute directly.
        database_hash = _hashes.get(resolved)
        if database_hash is None:
            direct = True
        else:
            direct = False
            if not _loaded:
                for row in read_jsonl(CACHE):
                    future = Future(); future.set_result(row["score"])
                    _futures[row["key"]] = future
                _loaded = True
            key = canonical_hash([resolved, database_hash, candidate_sql, gold_sql, timeout])
            owner = key not in _futures
            if owner:
                _futures[key] = Future()
            future = _futures[key]
    if direct:
        return execute_score(db_path, candidate_sql, gold_sql, timeout=timeout)
    if not owner:
        return dict(future.result())
    try:
        result = execute_score(db_path, candidate_sql, gold_sql, timeout=timeout)
        with _lock:
            if not result.get("infrastructure_error"):
                append_jsonl(CACHE, {"key": key, "database_sha256": database_hash,
                                    "original_sql": candidate_sql, "original_gold": gold_sql, "score": result})
            future.set_result(dict(result))
            if result.get("infrastructure_error"):
                _futures.pop(key, None)
        return result
    except BaseException as exc:
        with _lock:
            future.set_exception(exc); _futures.pop(key, None)
        raise
