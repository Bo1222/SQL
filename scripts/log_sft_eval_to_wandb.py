"""Log merged dev200 eval metrics to a W&B eval run.

The eval curve is a separate run (named ``<train_run_name>_eval``) with
``eval/step`` as its x-axis, because eval happens after training and must not
append out-of-order steps to the training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def metrics(path, base_ex):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ex = float(data.get("ex", 0.0))
    return {
        "eval/n": int(data.get("n", 0)),
        "eval/ex": ex,
        "eval/ex_correct": int(round(ex * int(data.get("n", 0)))),
        "eval/delta_ex_vs_base": ex - base_ex,
        "eval/valid": float(data.get("valid", 0.0)),
        "eval/lookup_rate": float((data.get("retrieval") or {}).get("lookup_rate", 0.0)),
        "eval/no_answer": int(data.get("no_answer", 0)),
        "eval/infra": int(data.get("infra", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--eval", action="append", required=True,
                        help="STEP=PATH, e.g. 394=reports/.../sft_lowlr_0394.json")
    parser.add_argument("--base-ex", type=float, default=0.495)
    parser.add_argument("--name", default=None,
                        help="eval W&B run name; default <train_name>_eval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = json.loads(Path(args.run_summary).read_text(encoding="utf-8"))
    info = summary.get("wandb") or {}
    rows = []
    for item in args.eval:
        if "=" not in item:
            raise SystemExit(f"--eval must be STEP=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label.isdigit():
            raise SystemExit(f"eval label must be the optimizer step, got {label!r}")
        rows.append((int(label), path, metrics(path, args.base_ex)))
    rows.sort()

    if args.dry_run:
        print(json.dumps([{"step": s, "path": p, **m} for s, p, m in rows],
                         ensure_ascii=False, indent=1))
        return
    run_id = info.get("run_id")
    if not run_id:
        raise SystemExit(f"no W&B run_id in {args.run_summary}: {info}")
    import wandb
    mode = info.get("mode", "online")
    # Eval happens after training; use a separate run with a custom step metric
    # so checkpoints can be logged at their training steps without violating the
    # training run's monotonic step history.
    run = wandb.init(
        project=info.get("project") or "sqlrl",
        name=args.name or f"{info.get('name', 'sft')}_eval",
        mode=mode,
        dir=str(Path(args.run_summary).parent),
        config={"train_run_id": run_id, "run_summary": args.run_summary},
    )
    wandb.define_metric("eval/step")
    for key in ("eval/ex", "eval/ex_correct", "eval/delta_ex_vs_base",
                "eval/valid", "eval/lookup_rate", "eval/no_answer",
                "eval/infra", "eval/n"):
        wandb.define_metric(key, step_metric="eval/step")
    for step, path, values in rows:
        run.log({"eval/step": step, **values})
        print(json.dumps({"step": step, "path": path, **values}, ensure_ascii=False))
    url = getattr(run, "url", None)
    eval_run_id = getattr(run, "id", None)
    run.finish()
    print(json.dumps({"train_run_id": run_id, "eval_run_id": eval_run_id,
                      "url": url, "logged": len(rows)}))


if __name__ == "__main__":
    main()
