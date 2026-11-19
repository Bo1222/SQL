"""Re-score saved valid-but-wrong predictions under two labelled comparators.

No model inference and no changes to existing EX/rewards. Official set-only:
https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1_alignment.common import read_jsonl, sha256_file, write_json
from tools.sql_execute import run_sql


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", action="append", required=True, help="NAME=JSONL")
    p.add_argument("--out", default="reports/m6/harness_audit_20261117/scoring_contract")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    started = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources, requests, membership = {}, {}, defaultdict(list)
    for value in a.trace:
        name, path = value.split("=", 1)
        rows = read_jsonl(path)
        selected = [r for r in rows if r.get("valid") and not r.get("correct")]
        sources[name] = {"path": path, "sha256": sha256_file(path), "n": len(rows),
                         "historical_correct": sum(bool(r.get("correct")) for r in rows),
                         "valid_but_wrong": len(selected)}
        for r in selected:
            request = (r["db_path"], r["final_sql"], r["gold_sql"])
            digest = hashlib.sha256(json.dumps(request).encode()).hexdigest()
            requests[digest] = request
            membership[digest].append({"trace": name, "id": r["id"], "db_id": r["db_id"]})
    path = out / "scores.jsonl"
    cached = {r["request_sha256"]: r["score"] for r in read_jsonl(path)}
    with ThreadPoolExecutor(max_workers=a.workers) as pool, path.open("a") as stream:
        jobs = {pool.submit(run_sql, req[0], req[1], operation="score_diagnostic", timeout=30,
                            gold_sql=req[2]): digest for digest, req in requests.items() if digest not in cached}
        for done, future in enumerate(as_completed(jobs), 1):
            digest = jobs[future]
            cached[digest] = future.result()
            stream.write(json.dumps({"request_sha256": digest, "score": cached[digest]}) + "\n")
            stream.flush()
            if done % 100 == 0:
                print(json.dumps({"scored": done, "jobs": len(jobs), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    cases = []
    for digest, members in membership.items():
        score = cached[digest]
        for member in members:
            cases.append({**member, "score": score})
    for name, summary in sources.items():
        selected = [r for r in cases if r["trace"] == name]
        summary.update({"rescored": len(selected),
            "strict_still_wrong": sum(r["score"].get("ok") and not r["score"].get("correct") for r in selected),
            "set_only_additional_correct": sum(r["score"].get("bird_set_only_correct", False) for r in selected),
            "reexecution_strict_flips": sum(bool(r["score"].get("correct")) for r in selected),
            "reexecution_errors": sum(not r["score"].get("ok") for r in selected)})
    write_json(out / "summary.json", {"diagnostic_only": True, "sources": sources,
        "official_source": "https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py",
        "worker_sha256": sha256_file("tools/sql_worker.py"),
        "scope": "Only historically valid-but-wrong predictions rescored. No historical EX or reward changed.",
        "differences": [r for r in cases if r["score"].get("bird_set_only_correct") or r["score"].get("correct")],
        "wall_seconds": round(time.monotonic() - started, 3)})
    print(json.dumps(sources), flush=True)


if __name__ == "__main__":
    main()
