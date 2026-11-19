"""Validate an M6-08m SFT JSONL before training.

Checks the stored token-level loss mask, context budget and (optionally) that
every record has trainable provenance.  Exit code 2 means the dataset is not
admissible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e1_alignment.student_sft_data import load_examples, summary, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--require-trainable", action="store_true")
    parser.add_argument("--out", default=None, help="optional JSON summary path")
    parser.add_argument("--excluded-out", default=None, help="optional excluded JSONL path")
    args = parser.parse_args()

    rows, excluded = load_examples(args.data, max_context=args.max_context,
                                   require_trainable=args.require_trainable)
    result = summary(rows, excluded)
    result["data"] = args.data
    result["require_trainable"] = args.require_trainable
    result["max_context"] = args.max_context
    result["ok"] = bool(rows) and not excluded
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
    if args.excluded_out and excluded:
        write_jsonl(args.excluded_out, excluded)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
