"""Rank-zero JSONL and optional W&B logging for GRPO."""
import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def training_logs(args, rank, world_size):
    if rank != 0:
        yield None, None
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    success = False
    with (out / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        try:
            if args.wandb_mode != "disabled":
                import wandb

                wandb_dir = Path(args.wandb_dir) if args.wandb_dir else out
                wandb_dir.mkdir(parents=True, exist_ok=True)
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    mode=args.wandb_mode,
                    dir=str(wandb_dir.resolve()),
                    config={**vars(args), "world_size": world_size},
                    settings=wandb.Settings(init_timeout=60),
                )
                run.define_metric("step")
                run.define_metric("*", step_metric="step")
                run.summary["status"] = "running"
                metadata = {
                    "id": run.id,
                    "project": run.project,
                    "mode": run.settings.mode,
                    "run_dir": str(Path(run.dir).parent),
                    "url": run.url if args.wandb_mode == "online" else None,
                }
                (out / "wandb_run.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[wandb] mode={metadata['mode']} id={run.id} dir={metadata['run_dir']}", flush=True)
            yield log_file, run
            success = True
        finally:
            if run is not None:
                run.summary["status"] = "completed" if success else "failed"
                run.finish(exit_code=0 if success else 1)


def log_metrics(log_file, run, record):
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_file.flush()
    if run is not None:
        run.log(record, step=record["step"])
