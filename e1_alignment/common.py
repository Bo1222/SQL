"""Small shared helpers for the E1 alignment pipeline."""
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL_TRAIN = ROOT / "data/processed/bird_agent/rl.jsonl"
RL_METADATA = ROOT / "data/processed/bird_agent/metadata"
DEV200 = ROOT / "artifacts/datasets/dev200.jsonl"
DEV200_MANIFEST = ROOT / "artifacts/manifests/dev200.json"
CATALOG_DIR = ROOT / "artifacts/catalog"
MODEL_PATH = ROOT / "artifacts/checkpoints/step_600_merged"
ORIGINAL_ADAPTER = ROOT / "checkpoints/bird_agent_full_sft_20261026/seed_0/step_600"


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def normalize_sql(sql):
    """Whitespace-normalized SQL for dedupe keys; preserves case and literals."""
    return re.sub(r"\s+", " ", str(sql or "").strip())


def stratified_sample(rows, limit, seed=0):
    """Deterministic proportional-by-db sample (same idea as eval.evaluate_agent)."""
    if not limit or limit >= len(rows):
        return list(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["db_id"]].append(row)
    for db in groups:
        groups[db].sort(key=lambda r: str(r["id"]))
    quotas = {db: int(len(v) * limit / len(rows)) for db, v in groups.items()}
    remainder = sorted(groups, key=lambda db: (-(len(groups[db]) * limit / len(rows) - quotas[db]), db))
    for db in remainder[: limit - sum(quotas.values())]:
        quotas[db] += 1
    rng = random.Random(seed)
    result = []
    for db in sorted(groups):
        result.extend(rng.sample(groups[db], quotas[db]))
    result.sort(key=lambda r: str(r["id"]))
    return result


def load_rows(path=RL_TRAIN):
    return read_jsonl(path)


def row_index(rows):
    return {row["id"]: row for row in rows}


def load_metadata(row):
    path = row.get("metadata_path") or RL_METADATA / f"{row['db_id']}.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_sql_cached(items, workers=8):
    """Score (db_path, sql, gold_sql) triples in parallel; returns keyed dict.

    Duplicate requests are collapsed.  The SQL worker runs in a subprocess, so
    threads only wait on independent processes.
    """
    from reward.bird_ex_reward import score_sql

    keys, seen = [], set()
    for db_path, sql, gold_sql in items:
        key = (str(db_path), normalize_sql(sql), normalize_sql(gold_sql))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    def run(key):
        db_path, sql, gold_sql = key
        try:
            return key, score_sql(db_path, sql, gold_sql, timeout=30)
        except Exception as exc:  # scoring must never kill a collection run
            return key, {"correct": False, "ok": False, "error": f"scorer failure: {exc}",
                         "error_type": "scorer_failure", "infrastructure_error": True}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(run, keys))


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_div(a, b):
    return a / b if b else 0.0


def quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = clamp((len(ordered) - 1) * q, 0, len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[int(index)]
    return ordered[low] * (high - index) + ordered[high] * (index - low)
