"""Paired EX comparison with exact McNemar and database-cluster bootstrap."""
import argparse
import math
import random
from collections import Counter, defaultdict

from agent.common import read_jsonl, write_json


def compare_records(a, b, bootstrap=2000, seed=0):
    A, B = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(A) != len(a) or len(B) != len(b) or set(A) != set(B):
        raise ValueError("Paired comparison requires identical unique IDs")
    if not A or bootstrap < 1:
        raise ValueError("nonempty records and positive bootstrap count required")
    gain, loss, clusters = [], [], defaultdict(list)
    for key in A:
        if any(A[key][field] != B[key][field] for field in ["db_id", "question", "gold_sql"]):
            raise ValueError("Mismatched task/gold for " + str(key))
        x, y = int(A[key]["final"]["correct"]), int(B[key]["final"]["correct"])
        if y > x: gain.append(key)
        if x > y: loss.append(key)
        clusters[A[key]["db_id"]].append(y - x)
    discordant = len(gain) + len(loss)
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(gain), len(loss)) + 1)) / 2 ** discordant) if discordant else 1.
    rng = random.Random(seed)
    dbs = sorted(clusters)
    samples = []
    for _ in range(bootstrap):
        differences = [d for db in rng.choices(dbs, k=len(dbs)) for d in clusters[db]]
        samples.append(sum(differences) / len(differences))
    samples.sort()
    return {"n": len(A), "dbs": len(dbs), "gain": len(gain), "loss": len(loss),
            "delta_ex": (len(gain) - len(loss)) / len(A), "mcnemar_exact_p": p,
            "db_cluster_ci95": [samples[int(.025 * (bootstrap - 1))], samples[int(.975 * (bootstrap - 1))]],
            "bootstrap": bootstrap, "seed": seed, "gain_ids": gain, "loss_ids": loss,
            "per_db_delta": {db: sum(v) / len(v) for db, v in clusters.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    write_json(a.out, compare_records(read_jsonl(a.a), read_jsonl(a.b), a.bootstrap, a.seed))


if __name__ == "__main__":
    main()
