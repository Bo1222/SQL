"""Audit provenance for M6-08m datasets.

Usage examples:

    python scripts/audit_provenance.py \
        --data artifacts/datasets/sft_trajectories.jsonl \
        --provenance-source data/processed/bird_agent/full_sft_20261026/full_tool_unique.jsonl \
        --out-dir reports/m6/e1_student_only/provenance_old

    python scripts/audit_provenance.py \
        --data artifacts/m6/e1_contract_rebuild/sft_trajectories_e1.jsonl \
        --tool-replay --out-dir reports/m6/e1_student_only/provenance_rebuild

    python scripts/audit_provenance.py \
        --data artifacts/m6/e1_student_only/train.jsonl \
        --mode trainable --out-dir reports/m6/e1_student_only/provenance_train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e1_alignment.provenance import (  # noqa: E402
    ProvenanceError,
    audit_records,
    read_jsonl,
    sha256_file,
    write_json,
)


def load_provenance_source(path):
    if not path:
        return {}
    source = {}
    for row in read_jsonl(path):
        key = row.get("id") or row.get("question_id")
        if key:
            source[key] = row
    return source


def mark_tool_replay(records):
    """Mark rebuilt tool-replay rows as non-trainable audit data."""
    for record in records:
        prov = record.get("provenance")
        if isinstance(prov, dict):
            prov["tool_replay"] = True
            prov["trainable"] = False
            continue
        if record.get("rebuild") is not None:
            record["_tool_replay"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True,
                        help="JSONL dataset; repeat for several files")
    parser.add_argument("--provenance-source", default=None,
                        help="old full_tool_unique.jsonl for legacy origin join")
    parser.add_argument("--tool-replay", action="store_true",
                        help="mark all records as tool replay (audit-only)")
    parser.add_argument("--mode", choices=("audit", "trainable"), default="audit")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_source = load_provenance_source(args.provenance_source)
    summary = {"datasets": [], "mode": args.mode}

    for data_path in args.data:
        records = list(read_jsonl(data_path))
        if args.tool_replay:
            for record in records:
                if record.get("rebuild") is not None:
                    record["_tool_replay"] = True
                prov = record.get("provenance")
                if isinstance(prov, dict):
                    prov["tool_replay"] = True
                    prov["trainable"] = False
        result = audit_records(records, provenance_source=provenance_source,
                               tool_replay=args.tool_replay)
        result["data"] = str(data_path)
        result["sha256"] = sha256_file(data_path)
        result["mode"] = args.mode
        result["tool_replay_records"] = sum(
            1 for r in records if r.get("_tool_replay") or (r.get("provenance") or {}).get("tool_replay")
        )
        summary["datasets"].append(result)

    summary["ok"] = True
    if args.mode == "trainable":
        for item in summary["datasets"]:
            if item["trainable_total"] != item["total"] or item["invalid_count"]:
                summary["ok"] = False
                summary["reason"] = (
                    f"{item['data']}: trainable={item['trainable_total']} total={item['total']} "
                    f"invalid={item['invalid_count']}"
                )
                break

    write_json(out_dir / "provenance_audit.json", summary)

    lines = ["# M6-08m provenance audit", "",
             f"- mode: `{args.mode}`",
             f"- ok: **{summary['ok']}**", ""]
    for item in summary["datasets"]:
        lines += [
            f"## `{item['data']}`",
            "",
            f"- sha256: `{item['sha256']}`",
            f"- total: **{item['total']}**",
            f"- trainable total: **{item['trainable_total']}**",
            f"- source counts: `{json.dumps(item['source_counts'], ensure_ascii=False)}`",
            f"- trainable counts: `{json.dumps(item['trainable_counts'], ensure_ascii=False)}`",
            f"- final_ex counts: `{json.dumps(item['final_ex_counts'], ensure_ascii=False)}`",
            f"- matched legacy provenance: {item['matched_legacy_provenance']}",
            f"- invalid: {item['invalid_count']}",
            f"- audit-only: {item['audit_only_count']}",
            f"- tool replay records: {item['tool_replay_records']}",
            "",
        ]
        if item["invalid"]:
            lines.append("Invalid examples:")
            for row in item["invalid"][:10]:
                lines.append(f"- `{row}`")
            lines.append("")
        if item["audit_only"]:
            lines.append("Audit-only examples:")
            for row in item["audit_only"][:10]:
                lines.append(f"- `{row}`")
            lines.append("")
    (out_dir / "provenance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
