"""Assess the SFT-compatibility diagnostic (A/B/C arms) on dev200.

Answers two questions:
  1. Does fixing the observation encoder change task/behavior metrics?
  2. Does forcing one proactive schema_lookup change them, and is the remaining
     bottleneck the SFT action-order prior or semantic SQL ability?
"""
import argparse
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mcnemar(a, b):
    gain = sum(1 for i in a if b.get(i) and not a.get(i))
    loss = sum(1 for i in a if a.get(i) and not b.get(i))
    n = gain + loss
    if n == 0:
        return gain, loss, 1.0
    k = min(gain, loss)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return gain, loss, p


def non_seeded(events):
    return [e for e in events if not e.get("seeded")]


def summarize(records, db_by_id=None):
    db_by_id = db_by_id or {}
    n = len(records)
    ex = sum(r["correct"] for r in records) / max(n, 1)
    valid = sum(r["valid"] for r in records) / max(n, 1)
    first_ex = 0
    repair_den = 0
    repair_ok = 0
    schema_errors = 0
    lookups = 0
    sql_executes = 0
    visible_lookup = 0
    visible_rel = 0
    fallback = 0
    first_action = Counter()
    recovery_template = 0
    ex_by_first = Counter()
    repair_by_rel = Counter()

    for r in records:
        events = r.get("tool_events", [])
        acts = non_seeded(events)
        first = acts[0]["action"] if acts else "none"
        first_action[first] += 1
        ex_by_first[(first, bool(r["correct"]))] += 1

        seq = tuple(e["action"] for e in acts)
        if seq == ("sql_execute", "schema_lookup", "sql_execute"):
            recovery_template += 1

        schema_lookups = [e for e in acts if e["action"] == "schema_lookup"]
        lookups += len(schema_lookups)
        sql_executes += sum(1 for e in acts if e["action"] == "sql_execute")

        exs = r.get("executions") or []
        first_error = bool(exs) and (exs[0].get("result") or {}).get("status") != "ok"
        if any((e.get("result") or {}).get("error_type") == "schema_error" for e in exs):
            schema_errors += 1
        if r.get("first_sql"):
            db_path = r.get("db_path") or db_by_id.get(r.get("id"))
            scored = score_sql(db_path, r["first_sql"], r.get("gold_sql"), timeout=30)
            first_ex += 1 if scored.get("correct") else 0
        if first_error:
            repair_den += 1
            if r["correct"]:
                repair_ok += 1

        vis = r.get("visible_lookup_observations", 0)
        rel = r.get("visible_relationship_observations", 0)
        visible_lookup += vis
        visible_rel += rel
        fallback += r.get("fallback_observations", 0)
        if first_error and schema_lookups:
            repair_by_rel[("relationships_visible" if rel else "no_relationships_visible", bool(r["correct"]))] += 1

    def rate(counter, key):
        yes = counter[(key, True)]
        total = counter[(key, True)] + counter[(key, False)]
        return yes / total if total else None

    return {
        "n": n,
        "ex": ex,
        "valid": valid,
        "first_sql_ex": first_ex / max(n, 1),
        "repair_cases": repair_den,
        "repair_success": repair_ok,
        "repair_rate": repair_ok / max(repair_den, 1),
        "schema_error_episodes": schema_errors,
        "avg_schema_lookups": lookups / max(n, 1),
        "avg_sql_executes": sql_executes / max(n, 1),
        "model_lookup_rate": sum(1 for r in records if any(e["action"] == "schema_lookup" for e in non_seeded(r.get("tool_events", [])))) / max(n, 1),
        "visible_lookup_observations": visible_lookup,
        "visible_relationship_observations": visible_rel,
        "visible_relationship_rate": visible_rel / max(visible_lookup, 1),
        "fallback_observations": fallback,
        "first_action": dict(first_action),
        "lookup_first_rate": first_action.get("schema_lookup", 0) / max(n, 1),
        "recovery_template_rate": recovery_template / max(n, 1),
        "ex_by_first_action": {
            key: {"correct": rate(ex_by_first, key),
                  "n": ex_by_first[(key, True)] + ex_by_first[(key, False)]}
            for key in ("schema_lookup", "sql_execute", "answer", "none")
        },
        "repair_by_visible_relationships": {
            key: {"correct": rate(repair_by_rel, key),
                  "n": repair_by_rel[(key, True)] + repair_by_rel[(key, False)]}
            for key in ("relationships_visible", "no_relationships_visible")
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="reports/m6")
    p.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    p.add_argument("--data", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--out", default="reports/m6/diag_sft_assessment.json")
    a = p.parse_args()

    db_by_id = {row["id"]: row["db_path"] for row in read_jsonl(a.data)}
    data = {}
    for arm in a.arms:
        path = Path(a.dir) / f"diag_sft_{arm}.jsonl"
        if path.is_file():
            data[arm] = read_jsonl(path)

    report = {"arms": {}, "paired": {}}
    for arm, records in data.items():
        report["arms"][arm] = summarize(records, db_by_id)

    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        if left not in data or right not in data:
            continue
        lmap = {r["id"]: bool(r["correct"]) for r in data[left]}
        rmap = {r["id"]: bool(r["correct"]) for r in data[right]}
        common = sorted(set(lmap) & set(rmap))
        gain, loss, pvalue = mcnemar({i: lmap[i] for i in common}, rmap)
        delta = (gain - loss) / max(len(common), 1)
        se = math.sqrt((gain + loss) / max(len(common), 1) ** 2)
        report["paired"][f"{left}_vs_{right}"] = {
            "n": len(common), "gain": gain, "loss": loss, "net": gain - loss,
            "delta_ex": delta,
            "ci95_delta": [round(delta - 1.96 * se, 4), round(delta + 1.96 * se, 4)],
            "mcnemar_p": round(pvalue, 4),
        }

    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
