"""Paired statistics between two evaluation record files (plan §12.6)."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mcnemar(gain, loss):
    discordant = gain + loss
    if not discordant:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2 * sum(math.comb(discordant, i) for i in range(k + 1)) / 2 ** discordant)


def bootstrap_ci(values_by_db, n=10000, seed=0):
    rng = random.Random(seed)
    keys = sorted(values_by_db)
    if not keys:
        return [None, None]
    samples = []
    for _ in range(n):
        picked = [v for db in rng.choices(keys, k=len(keys)) for v in values_by_db[db]]
        samples.append(sum(picked) / len(picked))
    samples.sort()
    return [samples[int(0.025 * (n - 1))], samples[int(0.975 * (n - 1))]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline records (E0)")
    p.add_argument("--b", required=True, help="treatment records (E1)")
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    left = {r["id"]: r for r in read_jsonl(a.a)}
    right = {r["id"]: r for r in read_jsonl(a.b)}
    common = sorted(set(left) & set(right))
    if not common:
        raise SystemExit("no paired questions")
    gain = [i for i in common if right[i]["correct"] and not left[i]["correct"]]
    loss = [i for i in common if left[i]["correct"] and not right[i]["correct"]]
    both = sum(1 for i in common if left[i]["correct"] and right[i]["correct"])
    neither = sum(1 for i in common if not left[i]["correct"] and not right[i]["correct"])
    by_db = defaultdict(list)
    for i in common:
        by_db[left[i]["db_id"]].append(int(bool(right[i]["correct"])) - int(bool(left[i]["correct"])))
    n = len(common)
    report = {
        "n_paired": n,
        "a_correct": sum(int(bool(left[i]["correct"])) for i in common),
        "b_correct": sum(int(bool(right[i]["correct"])) for i in common),
        "both_correct": both, "both_wrong": neither,
        "b_only": len(gain), "a_only": len(loss),
        "net_gain": len(gain) - len(loss),
        "delta_ex": (len(gain) - len(loss)) / n,
        "mcnemar_exact_p": mcnemar(len(gain), len(loss)),
        "db_clusters": len(by_db),
        "db_bootstrap_ci95": bootstrap_ci(by_db, a.bootstrap, a.seed),
        "gain_ids": gain, "loss_ids": loss,
    }
    for field in ("valid", "avg_latency_ms", "avg_response_tokens", "avg_observation_tokens",
                  "error_types", "retrieval", "avg_tool_calls", "avg_schema_lookups"):
        report[f"a_{field}"] = _agg(left, field)
        report[f"b_{field}"] = _agg(right, field)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_paired", "a_correct", "b_correct", "net_gain",
                                             "delta_ex", "mcnemar_exact_p", "db_bootstrap_ci95")},
                     ensure_ascii=False))


def _agg(records, field):
    values = list(records.values())
    if field == "valid":
        return sum(1 for r in values if r.get("valid")) / max(len(values), 1)
    if field == "error_types":
        counts = {}
        for r in values:
            key = str(r.get("error_type"))
            counts[key] = counts.get(key, 0) + 1
        return counts
    if field == "retrieval":
        keys = ("table_recall", "column_recall", "join_recall")
        return {k: sum((r.get("usage") or {}).get(k, 0) for r in values) / max(len(values), 1) for k in keys}
    if field == "avg_latency_ms":
        # Episode records store one latency per question as ``latency_ms``.
        return round(sum(r.get("latency_ms", 0) for r in values) / max(len(values), 1), 2)
    return round(sum(r.get(field, 0) for r in values) / max(len(values), 1), 2)


if __name__ == "__main__":
    main()
