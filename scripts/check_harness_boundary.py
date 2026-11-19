"""Offline checks of the exact gold paths that hit harness budgets.

The long-query rewrite shares an already present CTE expression; it is gold
assisted, diagnostic-only and must never be exported as training data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e1_alignment.common import read_jsonl, write_json
from eval.harness_audit import AuditTokenizer, gold_injection
from tools.sql_execute import run_sql


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="reports/m6/harness_audit_20261117/boundaries.json")
    a = p.parse_args()
    started = time.monotonic()
    rows = {r["id"]: r for r in read_jsonl("data/processed/bird_agent/test.jsonl")}
    tokenizer = AuditTokenizer("artifacts/checkpoints/step_600_merged")
    # Use the existing pinned parser dependency; do not install anything.
    sys.path.insert(0, str(ROOT / ".codex-deps/m6-08k"))
    import sqlglot
    from sqlglot import exp
    row = rows["bird_dev_1016"]
    tree = sqlglot.parse_one(row["gold_sql"], read="sqlite")
    ctes = {cte.alias: cte for cte in tree.find_all(exp.CTE)}
    lap = ctes["lap_record_race"].this
    lap.args["from_"].set("this", exp.Table(this=exp.to_identifier("fastest_lap_times"),
                                           alias=exp.TableAlias(this=exp.to_identifier("T1"))))
    lap.args["joins"][-1].set("on", exp.EQ(this=exp.column("time_in_seconds", table="T1"),
                                          expression=exp.column("min_time_in_seconds", table="T4")))
    rewritten = tree.sql(dialect="sqlite")
    new_row = {**row, "gold_sql": rewritten}
    rewrite = {"id": row["id"], "method": "reuse identical expression from existing fastest_lap_times CTE",
        "gold_assisted": True, "training_eligible": False,
        "original_action_tokens": len(tokenizer.encode("<answer>" + row["gold_sql"] + "</answer>")),
        "rewritten_action_tokens": len(tokenizer.encode("<answer>" + rewritten + "</answer>")),
        "rewritten_sql": rewritten, "submission": gold_injection(new_row, tokenizer),
        "score_against_original_gold": run_sql(row["db_path"], rewritten, operation="score", timeout=30,
                                                gold_sql=row["gold_sql"])}

    def long_check(qid):
        r = rows[qid]
        return {"id": qid, "timeout_seconds": 120,
                "score": run_sql(r["db_path"], r["gold_sql"], operation="score", timeout=120,
                                 gold_sql=r["gold_sql"])}

    with ThreadPoolExecutor(max_workers=2) as pool:
        timed_out_gold = list(pool.map(long_check, ["bird_dev_518", "bird_dev_701"]))
    report = {"diagnostic_only": True, "action_budget_rewrite": rewrite,
              "longer_timeout_gold_only": timed_out_gold,
              "limits": ["A passing rewrite is a witnessed valid alternative on this database, not a model result.",
                         "Longer-timeout scores are diagnostics; the production 30-second contract is unchanged."],
              "wall_seconds": round(time.monotonic() - started, 3)}
    write_json(a.out, report)
    print(report)


if __name__ == "__main__":
    main()
