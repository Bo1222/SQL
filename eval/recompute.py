"""Recompute metrics from saved raw outputs, without re-running the model."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402
from reward.exec_reward import audit_flags, compare, error_type, extract_sql  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="reports/base_eval_200.jsonl")
    parser.add_argument("--db-root", default="data/raw/dev_extract/dev_20240627/dev_databases")
    parser.add_argument("--out", default="reports/base_eval_200_recomputed.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = Counter()
    flags = Counter()
    info_flags = Counter()
    errors = Counter()
    difficulty = defaultdict(Counter)
    records = []

    for row in rows:
        db_path = row.get("db_path") or str(Path(args.db_root) / row["db_id"] / f"{row['db_id']}.sqlite")
        pred_sql, method = extract_sql(row.get("raw_output", ""))
        gold = execute(db_path, row["gold_sql"])
        record = dict(row)
        record.update(
            {
                "db_path": db_path,
                "pred_sql": pred_sql,
                "extract_method": method,
                "gold_ok": gold.ok,
                "gold_error": gold.error,
                "pred_ok": False,
                "pred_error": None,
                "correct": False,
                "flags": [],
                "info_flags": [],
                "pred_error_type": "none",
            }
        )
        stats["n"] += 1
        difficulty[row.get("difficulty", "")]["n"] += 1
        if gold.ok:
            stats["gold_ok"] += 1
        if pred_sql is None:
            stats["extract_missing"] += 1
            record["pred_error_type"] = "extract_missing"
        else:
            stats["extract_ok"] += 1
            pred = execute(db_path, pred_sql)
            record["pred_ok"] = pred.ok
            record["pred_error"] = pred.error
            if pred.ok:
                stats["pred_exec_ok"] += 1
                if gold.ok:
                    equal, reason = compare(gold.rows, pred.rows, row["gold_sql"])
                    record["compare_reason"] = reason
                    if equal:
                        stats["correct"] += 1
                        record["correct"] = True
                        difficulty[row.get("difficulty", "")]["correct"] += 1
                        hard, info = audit_flags(pred_sql, row["gold_sql"], gold.rows, pred.rows)
                        record["flags"] = hard
                        record["info_flags"] = info
                        for flag in hard:
                            flags[flag] += 1
                        for flag in info:
                            info_flags[flag] += 1
                        if hard:
                            stats["hack_correct"] += 1
            else:
                record["pred_error_type"] = error_type(pred.error)
                errors[record["pred_error_type"]] += 1
        records.append(record)

    summary = {
        "source_records": args.records,
        "n": stats["n"],
        "gold_ok": stats["gold_ok"],
        "gold_ok_rate": round(stats["gold_ok"] / stats["n"], 4),
        "extract_ok": stats["extract_ok"],
        "extract_ok_rate": round(stats["extract_ok"] / stats["n"], 4),
        "pred_exec_ok": stats["pred_exec_ok"],
        "pred_exec_ok_rate": round(stats["pred_exec_ok"] / stats["n"], 4),
        "correct": stats["correct"],
        "ex": round(stats["correct"] / stats["n"], 4),
        "correct_with_hard_flags": stats["hack_correct"],
        "ex_clean": round((stats["correct"] - stats["hack_correct"]) / stats["n"], 4),
        "hard_flags": dict(flags),
        "info_flags": dict(info_flags),
        "pred_error_types": dict(errors),
        "difficulty": {key: dict(value) for key, value in difficulty.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
