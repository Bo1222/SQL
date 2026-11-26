"""Compact progress view for the P0-3 learning check.

Reads the launcher log (`logs/m6_08m_rl/<run>.log`), the report status JSON,
the direction JSON, and the after-eval summary/log. It never starts a job.
"""
import argparse
from pathlib import Path
import json
import re

PROGRESS_RE = re.compile(
    r"Training Progress:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it\]")
STEP_RE = re.compile(r"step:(\d+)\s*-")
FIELDS = {
    "reward": r"critic/score/mean:([0-9.eE+-]+)",
    "entropy": r"actor/entropy:([0-9.eE+-]+)",
    "kl": r"actor/kl_loss:([0-9.eE+-]+)",
    "grad": r"actor/grad_norm:([0-9.eE+-]+)",
    "hbm": r"perf/max_memory_allocated_gb:([0-9.eE+-]+)",
    "step_s": r"timing_s/step:([0-9.eE+-]+)",
}
EVAL_RE = re.compile(r'\{"completed":\s*(\d+),\s*"of":\s*(\d+)')


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_step_metrics(log_path):
    path = Path(log_path)
    if not path.is_file():
        return {}
    last = None
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                if STEP_RE.search(line):
                    last = line
    except OSError:
        return {}
    if not last:
        return {}
    out = {}
    for name, pattern in FIELDS.items():
        match = re.search(pattern, last)
        if match:
            try:
                out[name] = float(match.group(1))
            except ValueError:
                pass
    return out


def progress_line(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    last = None
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                match = PROGRESS_RE.search(line)
                if match:
                    last = match
    except OSError:
        return None
    if not last:
        return None
    pct, current, total, elapsed, eta, sec_it = last.groups()
    return {"percent": int(pct), "current": int(current), "total": int(total),
            "elapsed": elapsed, "eta": eta, "sec_it": float(sec_it)}


def eval_progress(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    last = None
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                match = EVAL_RE.search(line)
                if match:
                    last = match
    except OSError:
        return None
    if not last:
        return None
    return {"completed": int(last.group(1)), "of": int(last.group(2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--after-log", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = Path(args.report_dir)
    status = read_json(report / "status.json") or {}
    phase = status.get("phase", "unknown")
    log_path = Path(args.log) if args.log else None
    if log_path is None and args.run_name:
        log_path = Path("logs/m6_08m_rl") / f"{args.run_name}.log"
    direction = read_json(report / "direction.json")
    after_summary = read_json(report / "after32" / "summary.json")
    after_log = Path(args.after_log) if args.after_log else report / "after32.log"
    out = {
        "phase": phase,
        "status": status,
        "log": str(log_path) if log_path else "",
        "progress": progress_line(log_path) if log_path else None,
        "last_step": latest_step_metrics(log_path) if log_path else {},
        "direction": direction,
        "after_summary": after_summary,
        "after_progress": eval_progress(after_log),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print(f"P0-3 phase: {phase}")
    if status.get("error"):
        print(f"ERROR: {status['error']}")
    if status.get("started_utc"):
        print(f"started_utc: {status['started_utc']}")
    prog = out["progress"]
    if prog:
        print(f"training: {prog['current']}/{prog['total']} ({prog['percent']}%) "
              f"elapsed={prog['elapsed']} eta={prog['eta']} ({prog['sec_it']:.1f}s/it)")
    last = out["last_step"]
    if last:
        print("last step: " + " ".join(
            f"{key}={value:.4g}" for key, value in last.items()))
    if direction:
        print(f"direction: pass={direction.get('pass_direction')} "
              f"delta_correct={direction.get('delta_correct'):.5g} "
              f"delta_incorrect={direction.get('delta_incorrect'):.5g} "
              f"gap={direction.get('gap_delta'):.5g}")
    if after_summary:
        print(f"after: complete={after_summary.get('complete')} "
              f"pass@1={after_summary.get('sampling_pass_at_1')} "
              f"pass@32={after_summary.get('sampling_pass_at_32')}")
    elif out["after_progress"]:
        ap = out["after_progress"]
        print(f"after: {ap['completed']}/{ap['of']} rollouts")
    print("")


if __name__ == "__main__":
    main()
