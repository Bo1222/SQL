#!/usr/bin/env python3
"""Parse veRL training log step lines into JSON/Markdown.

The launcher writes one line per optimizer step:

    step:1 - critic/score/mean:0.5 - ds/groups_kept:4 - ...

This utility is intentionally dependency-free and only reads the log; it does
not load models or touch NPUs.  For DAPO runs it also summarizes the
``ds/*`` fields emitted by ``train/dapo_ray_trainer.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")

# Metrics shown in the Markdown table; all parsed metrics remain in JSON.
TABLE_KEYS = [
    "timing_s/step",
    "critic/score/mean",
    "actor/grad_norm",
    "actor/kl_loss",
    "perf/max_memory_reserved_gb",
    "ds/num_gen_batches",
    "ds/groups_sampled",
    "ds/groups_kept",
    "ds/groups_filtered",
    "ds/groups_not_used",
    "ds/all_zero_groups",
    "ds/all_correct_groups",
    "ds/filter_rate",
    "ds/generated_tokens",
]


def _parse_value(text: str):
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def parse_log(path: str | Path) -> list[dict]:
    rows: dict[int, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw).replace("\r", "")
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            row: dict = {"step": step}
            for part in match.group(2).split(" - "):
                key, sep, value = part.partition(":")
                if not sep:
                    continue
                row[key] = _parse_value(value)
            rows[step] = row
    return [rows[step] for step in sorted(rows)]


def summarize(rows: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
        return out

    times = values("timing_s/step")
    rewards = values("critic/score/mean")
    hbm = values("perf/max_memory_reserved_gb")
    filter_rates = values("ds/filter_rate")
    summary = {
        "steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "step_time_mean_s": statistics.fmean(times) if times else None,
        "step_time_min_s": min(times) if times else None,
        "step_time_max_s": max(times) if times else None,
        "hbm_reserved_max_gb": max(hbm) if hbm else None,
    }
    if any("ds/num_gen_batches" in row for row in rows):
        summary.update({
            "filter_rate_mean": statistics.fmean(filter_rates) if filter_rates else None,
            "generated_tokens_total": int(sum(values("ds/generated_tokens"))),
            "gen_batches_total": int(sum(values("ds/num_gen_batches"))),
            "underfilled_steps": int(
                sum(1 for row in rows if row.get("ds/underfilled"))
            ),
        })
    return summary


def write_markdown(rows: list[dict], summary: dict, path: str | Path, title: str) -> None:
    lines = [f"# {title}", "", "## Step metrics", ""]
    header = ["step"] + TABLE_KEYS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        cells = [str(row.get("step", ""))]
        for key in TABLE_KEYS:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--title", default="RL training summary")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"no step lines found in {args.log}")
    summary = summarize(rows)
    Path(args.out_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.out_md:
        write_markdown(rows, summary, args.out_md, args.title)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
