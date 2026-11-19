"""Persistent, conservative NPU budget and process-group wall-clock watchdog."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from e1_alignment.common import write_json

LIMIT = 4 * 3600
COUNTS = {"A": 800, "B": 600, "C": 200, "train": 1, "monitor_sft": 1,
          "monitor_candidate": 1, "dev200_candidate": 1, "recovery_pair": 1}


@contextmanager
def locked(path):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def reserve(path, stage, seconds, count, command):
    if stage not in COUNTS or seconds <= 0 or count <= 0:
        raise ValueError("invalid_budget_request")
    if stage == "train" and seconds > 3300:
        raise ValueError("training_exceeds_55_minutes")
    reservation = seconds + (30 if stage == "train" else 5)
    with locked(path):
        state = json.loads(Path(path).read_text()) if Path(path).exists() else {"limit_s": LIMIT, "jobs": []}
        # Running or orphaned jobs cost their whole reservation until finalized.
        if any(j["status"] == "running" for j in state["jobs"]):
            raise ValueError("previous_job_running_or_unreconciled")
        charged = sum(j.get("charged_s", j["reserved_s"]) for j in state["jobs"])
        used = sum(j.get("consumed_count", j["count"]) for j in state["jobs"] if j["stage"] == stage)
        if charged + reservation > LIMIT or used + count > COUNTS[stage]:
            raise ValueError("budget_exhausted")
        job = {"id": len(state["jobs"]), "stage": stage, "count": count, "reserved_s": reservation, "timeout_s": seconds,
               "status": "running", "command": command, "started_at": time.time()}
        state["jobs"].append(job)
        write_json(path, state)
        return job["id"]


def finish(path, job_id, elapsed, code, timed_out):
    with locked(path):
        state = json.loads(Path(path).read_text())
        job = state["jobs"][job_id]
        # The collector creates its raw journal before entering the sampling loop.
        # A failure with no journal is provably before the first sampling attempt.
        raw = Path(f"artifacts/m6/sft_rebuild/new_{job['stage']}.raw.jsonl")
        if job["stage"] in ("A", "B", "C") and code != 0 and not raw.exists():
            job["consumed_count"] = 0
            job["count_evidence"] = "collector failed before raw journal creation and sampling loop"
        job.update(status="timeout" if timed_out else ("complete" if code == 0 else "failed"),
                   charged_s=elapsed, actual_wall_s=elapsed,
                   exit_code=code, finished_at=time.time())
        write_json(path, state)


def active_budget(stage):
    path = os.environ.get("SQLRL_REBUILD_BUDGET_LEDGER")
    index = os.environ.get("SQLRL_REBUILD_BUDGET_JOB")
    if not path or index is None:
        raise ValueError("must_run_via_persistent_budget_watchdog")
    state = json.loads(Path(path).read_text())
    job = state["jobs"][int(index)]
    if job["stage"] != stage or job["status"] != "running":
        raise ValueError("budget_receipt_not_active_for_stage")
    remaining = job.get("timeout_s", job["reserved_s"]) - (time.time() - job["started_at"])
    if remaining <= 0:
        raise ValueError("job_budget_already_expired")
    return remaining


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default="reports/m6/sft_rebuild/npu_budget.json")
    p.add_argument("--stage", choices=COUNTS, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command:
        raise SystemExit("missing command")
    job_id = reserve(a.ledger, a.stage, a.seconds, a.count, command)
    started = time.monotonic()
    code, timed_out = 1, False
    try:
        process = subprocess.Popen(command, start_new_session=True,
                                   env={**os.environ, "SQLRL_REBUILD_BUDGET_JOB": str(job_id),
                                        "SQLRL_REBUILD_BUDGET_LEDGER": str(Path(a.ledger).resolve())})
        try:
            code = process.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Trainer's SIGALRM handles state saving. Other jobs terminate immediately.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30 if a.stage == "train" else 5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    finally:
        finish(a.ledger, job_id, time.monotonic() - started, code, timed_out)
    raise SystemExit(code if not timed_out else 124)


if __name__ == "__main__":
    main()
