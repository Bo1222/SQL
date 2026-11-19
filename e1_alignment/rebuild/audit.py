"""Audit old/new candidates, freeze a genuinely unused train-domain monitor set.

No semantic approvals are synthesized. Passing program checks creates pending
individual-review records. CLI progress is an audit count, never task ability.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from e1_alignment.common import (RL_TRAIN, canonical_hash, read_jsonl, stratified_sample,
                                 sha256_file, write_json, write_jsonl)
from e1_alignment.rebuild.freeze import distribution
from e1_alignment.rebuild.quality import program_check, versioned
from e1_alignment.runtime import Runtime, RuntimeContract

ARTIFACT = Path("artifacts/m6/sft_rebuild")
REPORT = Path("reports/m6/sft_rebuild")
OLD = Path("artifacts/m6/e1_alignment")
DEFAULT_CANDIDATES = [OLD / f for f in ("e1_self_rollouts.jsonl", "e1_repair_rollouts.jsonl", "e1_gold_assisted.jsonl")]


def iter_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def make_monitor(train, path):
    if Path(path).exists():
        rows = read_jsonl(path)
        if len(rows) != 64:
            raise ValueError("invalid_existing_monitor")
        return rows
    used = set()
    sources = list(OLD.glob("*.jsonl"))
    sources += sorted(Path("reports/m6").glob("traces*/**/rollouts*.jsonl"))
    for file in sources:
        for row in iter_rows(file):
            used.add(row.get("question_id") or row.get("id"))
    available = [r for r in train.values() if r["id"] not in used]
    rows = stratified_sample(available, 64, 0)
    if len(rows) != 64:
        raise ValueError("not_enough_unused_monitor_questions")
    write_jsonl(path, rows)
    write_json(REPORT / "monitor_manifest.json", {"seed": 0, "n": len(rows),
               "sha256": sha256_file(path), "available_unused": len(available),
               "source_files": {str(p): sha256_file(p) for p in sources},
               "excluded_used_question_ids_sha256": canonical_hash(sorted(x for x in used if x)),
               "rule": "RL Train only; no prior branch collection or rollout trace question"})
    return rows


def verified_prefixes(paths):
    """A saved failure manifest is a pointer, not proof of a policy failure."""
    result, rejected = {}, []
    cache = {}
    for path in paths:
        for prefix in read_jsonl(path):
            origin = prefix.get("source_trace") or {}
            if origin.get("dir") == "source_sft":
                source = Path("artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl")
            else:
                source = Path("reports/m6/traces_lr_diag") / str(origin.get("dir")) / str(origin.get("file"))
            if source not in cache:
                cache[source] = read_jsonl(source)
            found = False
            for trace in cache[source]:
                if (trace.get("question_id") or trace.get("id")) != prefix.get("question_id"):
                    continue
                if trace.get("final_sql") != origin.get("trace_final_sql"):
                    continue
                n = prefix.get("prefix_tool_calls", 0)
                expected = [{k: event.get(k) for k in ("action", "arguments", "result")}
                            for event in trace.get("tool_events", [])[:n]]
                actual = [{k: event.get(k) for k in ("action", "arguments", "result")}
                          for event in prefix.get("tool_events", [])]
                if not n or canonical_hash(expected) != canonical_hash(actual):
                    continue
                if origin.get("origin_sha256"):
                    normalized = {**trace, "question_id": trace.get("id"),
                                  "_source_file": source.name, "_source_dir": "source_sft"}
                    if canonical_hash(normalized) != origin["origin_sha256"]:
                        continue
                    true_segments = trace.get("segments", [])[1:1 + len(prefix.get("segments", []))]
                    if [(s.get("text"), s.get("ids")) for s in true_segments] != [
                            (s.get("text"), s.get("ids")) for s in prefix.get("segments", [])]:
                        continue
                found = True
                break
            if found:
                result.setdefault(prefix["state_id"], []).append(prefix)
            else:
                rejected.append({"state_id": prefix.get("state_id"), "source": str(source),
                                 "reason": "failure_origin_not_verified"})
    return result, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", nargs="+", default=[str(f) for f in DEFAULT_CANDIDATES])
    p.add_argument("--train", default=str(RL_TRAIN))
    p.add_argument("--monitor", default=str(ARTIFACT / "monitor64.jsonl"))
    p.add_argument("--prefixes", nargs="+", default=[str(OLD / "e1_failure_prefixes.jsonl")])
    p.add_argument("--out", default=str(ARTIFACT / "audited_candidates.jsonl"))
    p.add_argument("--ledger", default=str(REPORT / "quality_ledger.jsonl"))
    p.add_argument("--summary", default=str(REPORT / "audit_summary.json"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    started = time.monotonic()
    train = {r["id"]: r for r in read_jsonl(a.train)}
    monitor = make_monitor(train, a.monitor)
    fingerprint = json.loads(Path("reports/m6/e1_alignment/runtime_fingerprint.json").read_text())["fingerprint_sha256"]
    runtime = Runtime(RuntimeContract())
    prefixes, unverified_prefixes = verified_prefixes(a.prefixes)
    rows = [versioned(r, path, i) for path in a.candidates for i, r in enumerate(read_jsonl(path), 1)]
    # Warm caches before thread workers share read-only tools.
    for db in sorted({r["db_id"] for r in rows}):
        runtime.schema_tool(db)
    forbidden = set()
    for name in ("val", "test"):
        for r in read_jsonl(Path("data/processed/bird_agent") / f"{name}.jsonl"):
            forbidden.add(r["db_id"])

    def check(row):
        row["program_check"] = program_check(row, train, {r["id"] for r in monitor}, fingerprint,
                                              runtime=runtime, forbidden_dbs=forbidden,
                                              prefix_index=prefixes)
        row["ast_check"] = row["program_check"]["ast"]
        row["visible_coverage"] = row["program_check"]["visible_coverage"]
        row["join_verification"] = {"status": "pending_individual_semantic_review",
                                     "ast_bound": row["program_check"]["ast"].get("gold", {}).get("joins", [])}
        return row

    checked = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(check, rows):
            checked.append(row)
            if len(checked) % 25 == 0:
                print(json.dumps({"audited": len(checked), "program_passed": sum(r["program_check"]["ok"] for r in checked),
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    write_jsonl(a.out, checked)
    ledger = []
    for r in checked:
        check = r["program_check"]
        ledger.append({"trajectory_id": r["trajectory_id"], "id": r["id"], "db_id": r["db_id"],
                       "source": r["trajectory_source"], "origin": r["origin"],
                       "content_sha256": r["content_sha256"], "parent_version_id": r["parent_version_id"],
                       "program_passed": check["ok"], "reasons": check["reasons"], "warnings": check["warnings"],
                       "status": "pending_individual_review" if check["ok"] else "excluded",
                       "semantic_review": None})
    write_jsonl(a.ledger, ledger)
    good = [r for r in checked if r["program_check"]["ok"]]
    summary = {"stage": "M6-08j-audit", "measurement": "实测", "candidates": len(checked),
               "program_passed": len(good), "semantic_approved": 0, "frozen": False, "unverified_failure_prefixes": unverified_prefixes,
               "rejections": dict(Counter(x for r in checked for x in r["program_check"]["reasons"])),
               "warnings": dict(Counter(x for r in checked for x in r["program_check"]["warnings"])),
               "all_distribution": distribution(checked), "program_pass_distribution": distribution(good),
               "elapsed_s": round(time.monotonic() - started, 1), "candidates_out": a.out,
               "ledger": a.ledger, "monitor_sha256": sha256_file(a.monitor),
               "review_note": "Program checks cannot approve semantics; pending rows require actual individual reading."}
    write_json(a.summary, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
