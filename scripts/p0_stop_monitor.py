"""Enforcing stop monitor for a P0-3-style GRPO run.

Reads the launcher log, evaluates each new optimizer step, and kills the
launcher's process group when a pre-registered stop condition triggers. It is
intentionally independent of the training loop so a violated stop rule cannot
be ignored.

Stop conditions (defaults):
  actor/kl_loss > 0.01
  actor/entropy < 0.01
  actor/grad_norm == 0 or > 10 for 3 consecutive steps
  perf/max_memory_allocated_gb > 30
  timing_s/step > 720 s (12 min; calibrated for the 4-prompt batch)
  ds/reward_unavailable_groups > 0 or ds/infra_groups > 0
  any non-finite metric
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import time

STEP_RE = re.compile(r"step:(\d+)\s*-")
FIELDS = {
    "reward": r"critic/score/mean:([0-9.eE+-]+)",
    "entropy": r"actor/entropy:([0-9.eE+-]+)",
    "kl_loss": r"actor/kl_loss:([0-9.eE+-]+)",
    "grad_norm": r"actor/grad_norm:([0-9.eE+-]+)",
    "hbm_gb": r"perf/max_memory_allocated_gb:([0-9.eE+-]+)",
    "step_s": r"timing_s/step:([0-9.eE+-]+)",
    "unavailable_groups": r"ds/reward_unavailable_groups:([0-9]+)",
    "infra_groups": r"ds/infra_groups:([0-9]+)",
}


def parse_last(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    last = None
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                if STEP_RE.search(line):
                    last = line
    except OSError:
        return None
    if not last:
        return None
    match = STEP_RE.search(last)
    step = int(match.group(1))
    metrics = {"step": step}
    for name, pattern in FIELDS.items():
        found = re.search(pattern, last)
        if found:
            try:
                metrics[name] = float(found.group(1))
            except ValueError:
                metrics[name] = found.group(1)
    if re.search(r"(?<![a-z])(nan|inf|-inf)(?![a-z])", last.lower()):
        metrics["non_finite_text"] = True
    return metrics


def nonfinite(value):
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def stop_reason(metrics, state, args):
    if metrics.get("non_finite_text"):
        return "non-finite metric text"
    for key, value in metrics.items():
        if nonfinite(value):
            return f"non-finite {key}"
    kl = metrics.get("kl_loss")
    if isinstance(kl, float) and kl > args.kl_max:
        return f"KL loss {kl:.6g} > {args.kl_max:g}"
    entropy = metrics.get("entropy")
    if isinstance(entropy, float) and entropy < args.entropy_min:
        return f"entropy {entropy:.6g} < {args.entropy_min:g}"
    grad = metrics.get("grad_norm")
    if isinstance(grad, float):
        if grad == 0.0 or grad > args.grad_max:
            state["grad_streak"] += 1
        else:
            state["grad_streak"] = 0
        if state["grad_streak"] >= args.grad_consecutive:
            return f"grad_norm {grad:.6g} outside (0,{args.grad_max:g}] for {state['grad_streak']} steps"
    hbm = metrics.get("hbm_gb")
    if isinstance(hbm, float) and hbm > args.hbm_max:
        return f"HBM {hbm:.3f} GB > {args.hbm_max:g} GB"
    step_s = metrics.get("step_s")
    if isinstance(step_s, float) and step_s > args.step_seconds_max:
        return f"step time {step_s:.1f}s > {args.step_seconds_max:g}s"
    for key, label in (("unavailable_groups", "unavailable reward groups"),
                       ("infra_groups", "infrastructure groups")):
        value = metrics.get(key)
        if isinstance(value, float) and value > 0:
            return f"{label}: {value:g}"
    return None


def kill_group(pgid, grace_s):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--kl-max", type=float, default=0.01)
    parser.add_argument("--entropy-min", type=float, default=0.01)
    parser.add_argument("--grad-max", type=float, default=10.0)
    parser.add_argument("--grad-consecutive", type=int, default=3)
    parser.add_argument("--hbm-max", type=float, default=30.0)
    parser.add_argument("--step-seconds-max", type=float, default=720.0)
    parser.add_argument("--kill-grace-s", type=float, default=20.0)
    args = parser.parse_args()
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    stop_path = report / "stop_trigger.json"
    status_path = report / "monitor_status.json"
    state = {"grad_streak": 0, "last_step": None}

    def evaluate(metrics):
        if not metrics or metrics["step"] == state["last_step"]:
            return None
        state["last_step"] = metrics["step"]
        reason = stop_reason(metrics, state, args)
        status_path.write_text(json.dumps(
            {"monitor": "running", "metrics": metrics, "reason": reason,
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2) + "\n")
        return reason

    while True:
        if stop_path.exists():
            return 0
        # Evaluate a newly written step before checking whether the launcher
        # exited: the final step line may arrive just before process exit.
        reason = evaluate(parse_last(args.log))
        if reason:
            stop_path.write_text(json.dumps(
                {"stopped": True, "reason": reason, "metrics": parse_last(args.log),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                indent=2) + "\n")
            kill_group(args.pgid, args.kill_grace_s)
            return 2
        try:
            os.killpg(args.pgid, 0)
        except ProcessLookupError:
            # One final evaluation after the process group is gone.
            reason = evaluate(parse_last(args.log))
            if reason:
                stop_path.write_text(json.dumps(
                    {"stopped": True, "reason": reason, "metrics": parse_last(args.log),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                    indent=2) + "\n")
                return 2
            status_path.write_text(json.dumps(
                {"monitor": "exited", "reason": "process group gone",
                 "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                indent=2) + "\n")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
