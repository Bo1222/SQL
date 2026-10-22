"""Zero-shot base evaluation: EX, valid execution rate, audit flags, error types."""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402
from reward.exec_reward import audit_flags, compare, error_type, extract_sql  # noqa: E402
from rollout.hf_rollout import HFRollout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/spider/test_200.jsonl")
    parser.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--partial-out", default="")
    parser.add_argument("--out", default="reports/spider_base200.json")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    print(
        f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} data={args.data} n={len(rows)} "
        f"batch={args.batch_size} max_new={args.max_new_tokens} model={args.model}",
        flush=True,
    )

    started = time.time()
    rollout = HFRollout(args.model, max_new_tokens=args.max_new_tokens, adapter=args.adapter)
    print(f"[load] model loaded in {time.time() - started:.1f}s", flush=True)

    def on_generate(done, total, seconds):
        if done % max(args.log_every, 1) == 0 or done == total:
            print(f"[gen] {done}/{total} elapsed={seconds:.1f}s", flush=True)

    predictions = rollout.generate(
        [row["prompt"] for row in rows],
        batch_size=args.batch_size,
        temperature=args.temperature,
        progress=on_generate,
    )
    print(f"[gen] done {len(predictions)} predictions in {time.time() - started:.1f}s", flush=True)

    partial_path = Path(args.partial_out) if args.partial_out else Path(args.out).with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path.open("w", encoding="utf-8")

    stats = Counter()
    flags = Counter()
    info_flags = Counter()
    errors = Counter()
    difficulty = defaultdict(Counter)
    records = []

    for idx, (row, text) in enumerate(zip(rows, predictions), start=1):
        pred_sql, method = extract_sql(text)
        gold = execute(row["db_path"], row["gold_sql"])
        record = {
            "id": row["id"],
            "db_id": row["db_id"],
            "db_path": row["db_path"],
            "difficulty": row.get("difficulty", ""),
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "pred_sql": pred_sql,
            "extract_method": method,
            "raw_output": text,
            "gold_ok": gold.ok,
            "gold_error": gold.error,
            "pred_ok": False,
            "pred_error": None,
            "correct": False,
            "flags": [],
            "info_flags": [],
            "pred_error_type": "none",
        }

        stats["n"] += 1
        difficulty[row.get("difficulty", "")]["n"] += 1
        if gold.ok:
            stats["gold_ok"] += 1
        if pred_sql is None:
            stats["extract_missing"] += 1
            record["pred_error_type"] = "extract_missing"
        else:
            stats["extract_ok"] += 1
            pred = execute(row["db_path"], pred_sql)
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
        partial.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial.flush()

        if idx % max(args.log_every, 1) == 0 or idx == len(rows):
            elapsed = time.time() - started
            ex_rate = stats["correct"] / idx
            valid_rate = stats["pred_exec_ok"] / idx
            print(
                f"[score] {idx}/{len(rows)} ex={stats['correct']} ({ex_rate:.1%}) "
                f"valid={stats['pred_exec_ok']} ({valid_rate:.1%}) elapsed={elapsed:.0f}s "
                f"last={row['id']}",
                flush=True,
            )

    partial.close()
    elapsed = time.time() - started
    summary = {
        "data": args.data,
        "model": args.model,
        "adapter": args.adapter,
        "limit": len(rows),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "elapsed_s": round(elapsed, 1),
        "n": stats["n"],
        "gold_ok": stats["gold_ok"],
        "gold_ok_rate": round(stats["gold_ok"] / stats["n"], 4),
        "extract_ok": stats["extract_ok"],
        "extract_ok_rate": round(stats["extract_ok"] / stats["n"], 4),
        "pred_exec_ok": stats["pred_exec_ok"],
        "pred_exec_ok_rate": round(stats["pred_exec_ok"] / stats["n"], 4),
        "correct": stats["correct"],
        "ex": round(stats["correct"] / stats["n"], 4),
        "ex_over_gold_ok": round(stats["correct"] / stats["gold_ok"], 4) if stats["gold_ok"] else 0.0,
        "correct_with_hard_flags": stats["hack_correct"],
        "ex_clean": round((stats["correct"] - stats["hack_correct"]) / stats["n"], 4),
        "hard_flags": dict(flags),
        "info_flags": dict(info_flags),
        "pred_error_types": dict(errors),
        "difficulty": {key: dict(value) for key, value in difficulty.items()},
        "s_per_sample": round(elapsed / max(len(rows), 1), 2),
        "partial_records": str(partial_path),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {out} and {out.with_suffix('.jsonl')}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
