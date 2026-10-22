"""Estimate pass@k / group variance on a prompt subset (no training)."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sqlite_exec import execute  # noqa: E402
from reward.exec_reward import compare, extract_sql  # noqa: E402
from rollout.hf_rollout import HFRollout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/spider/rl_100.jsonl")
    parser.add_argument("--model", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default="reports/pass8_rl100.json")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    started = time.time()
    print(f"[start] n_prompts={len(rows)} n={args.n} data={args.data} adapter={args.adapter}", flush=True)

    rollout = HFRollout(args.model, max_new_tokens=args.max_new_tokens, adapter=args.adapter)
    print(f"[load] {time.time() - started:.1f}s", flush=True)

    def on_progress(done, total, seconds):
        if done % max(args.log_every, 1) == 0 or done == total:
            print(f"[gen] {done}/{total} elapsed={seconds:.1f}s", flush=True)

    outputs = rollout.generate(
        [row["prompt"] for row in rows],
        batch_size=args.batch_size,
        temperature=args.temperature,
        n=args.n,
        progress=on_progress,
    )

    distribution = Counter()
    records = []
    total_samples = total_valid = total_correct = total_extract = 0
    pass_count = 0
    mixed = full = zero = 0

    for row, samples in zip(rows, outputs):
        gold = execute(row["db_path"], row["gold_sql"])
        correct = valid = 0
        for text in samples:
            total_samples += 1
            sql, _ = extract_sql(text)
            if sql is None:
                continue
            total_extract += 1
            pred = execute(row["db_path"], sql)
            if not pred.ok:
                continue
            valid += 1
            total_valid += 1
            if gold.ok and compare(gold.rows, pred.rows, row["gold_sql"])[0]:
                correct += 1
                total_correct += 1
        distribution[correct] += 1
        pass_count += 1 if correct > 0 else 0
        if correct == 0:
            zero += 1
        elif correct == args.n:
            full += 1
        else:
            mixed += 1
        records.append(
            {
                "id": row["id"],
                "db_id": row["db_id"],
                "question": row["question"],
                "gold_sql": row["gold_sql"],
                "n": args.n,
                "correct": correct,
                "valid": valid,
                "pass_at_n": 1 if correct > 0 else 0,
            }
        )
        if len(records) % max(args.log_every, 1) == 0:
            print(
                f"[score] {len(records)}/{len(rows)} pass@{args.n}={pass_count / len(records):.2%} "
                f"mixed={mixed / len(records):.1%} elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    summary = {
        "data": args.data,
        "adapter": args.adapter,
        "n_prompts": len(rows),
        "n": args.n,
        "temperature": args.temperature,
        "elapsed_s": round(time.time() - started, 1),
        "pass_at_n": round(pass_count / max(len(rows), 1), 4),
        "mean_correct_per_prompt": round(total_correct / max(len(rows), 1), 3),
        "mixed_group_ratio": round(mixed / max(len(rows), 1), 4),
        "all_zero_ratio": round(zero / max(len(rows), 1), 4),
        "all_correct_ratio": round(full / max(len(rows), 1), 4),
        "valid_rate": round(total_valid / max(total_samples, 1), 4),
        "extract_rate": round(total_extract / max(total_samples, 1), 4),
        "distribution": {str(k): v for k, v in sorted(distribution.items())},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
