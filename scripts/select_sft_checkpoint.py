"""Select an SFT epoch checkpoint using dev200; test set is never consulted."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.paired_statistics import bootstrap_ci, mcnemar  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load(path):
    rows = read_jsonl(path)
    return {row["id"]: row for row in rows}


def stats(records, base=None):
    values = list(records.values())
    n = len(values)
    lookup = [r for r in values if (r.get("usage") or {}).get("looked_up")]
    result = {
        "n": n,
        "ex": sum(1 for r in values if r.get("correct")) / max(n, 1),
        "correct": sum(1 for r in values if r.get("correct")),
        "valid": sum(1 for r in values if r.get("valid")) / max(n, 1),
        "no_answer": sum(1 for r in values if r.get("finish_reason") != "answer_submitted"),
        "infra": sum(1 for r in values if r.get("infrastructure_error")),
        "lookup_rate": len(lookup) / max(n, 1),
        "empty_lookup_rate": (
            sum(1 for r in lookup
                if not ((r.get("usage") or {}).get("tables") or [])) / max(len(lookup), 1)),
        "avg_tool_calls": sum(r.get("tool_calls", 0) for r in values) / max(n, 1),
        "error_types": dict(Counter(str(r.get("error_type")) for r in values)),
    }
    if base is not None:
        common = sorted(set(base) & set(records))
        if len(common) != len(base) or len(common) != len(records):
            raise SystemExit(
                f"paired ids differ: base={len(base)} epoch={len(records)} common={len(common)}")
        gain = [i for i in common if records[i].get("correct") and not base[i].get("correct")]
        loss = [i for i in common if base[i].get("correct") and not records[i].get("correct")]
        by_db = {}
        for i in common:
            db = base[i]["db_id"]
            by_db.setdefault(db, []).append(
                int(bool(records[i].get("correct"))) - int(bool(base[i].get("correct"))))
        result.update({
            "base_correct": sum(1 for i in common if base[i].get("correct")),
            "gain": len(gain), "loss": len(loss), "net_gain": len(gain) - len(loss),
            "delta_ex": (len(gain) - len(loss)) / max(len(common), 1),
            "mcnemar_exact_p": mcnemar(len(gain), len(loss)),
            "db_bootstrap_ci95": bootstrap_ci(by_db, n=10000, seed=0),
            "gain_ids": gain, "loss_ids": loss,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="dev200 base records JSONL (reports/.../dev200_base_e1_1.jsonl)")
    parser.add_argument("--epoch", action="append", required=True,
                        help="EPOCH=PATH, repeatable, e.g. 1=reports/.../sft_full_epoch1.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    base = load(args.base)
    base_stats = stats(base)
    epochs = {}
    for item in args.epoch:
        if "=" not in item:
            raise SystemExit(f"--epoch must be EPOCH=PATH, got {item!r}")
        label, path = item.split("=", 1)
        epochs[label] = {"path": path, "records": load(path)}
        epochs[label]["stats"] = stats(epochs[label]["records"], base)

    order = {label: index for index, label in enumerate(epochs)}
    ranked = sorted(epochs, key=lambda label: (-epochs[label]["stats"]["ex"],
                                               order[label]))
    best = ranked[0]
    best_stats = epochs[best]["stats"]
    warnings = []
    if best_stats["valid"] < base_stats["valid"] - 0.05:
        warnings.append(
            f"valid collapsed: {base_stats['valid']:.3f} -> {best_stats['valid']:.3f}")
    if best_stats["lookup_rate"] < base_stats["lookup_rate"] - 0.10:
        warnings.append(
            f"lookup rate collapsed: {base_stats['lookup_rate']:.3f} -> "
            f"{best_stats['lookup_rate']:.3f}")

    payload = {
        "selection_data": args.base,
        "selection_rule": "max dev200 EX; tie -> earlier epoch",
        "base": base_stats,
        "epochs": {label: {"path": value["path"], **value["stats"]}
                   for label, value in epochs.items()},
        "ranking": ranked,
        "best_epoch": best,
        "warnings": warnings,
        "note": "dev200 is a subset of val590; do not report val590 as an independent "
                "test. Use test1534 for the final held-out comparison.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = [
        "# SFT checkpoint selection (dev200)", "",
        f"- rule: {payload['selection_rule']}",
        f"- base: EX {base_stats['ex']:.3f} ({base_stats['correct']}/{base_stats['n']}), "
        f"valid {base_stats['valid']:.3f}, lookup {base_stats['lookup_rate']:.3f}",
        "",
        "| epoch | EX | delta EX | gain/loss | McNemar p | valid | lookup | no_answer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ranked:
        item = epochs[label]["stats"]
        lines.append(
            f"| {label} | {item['ex']:.3f} ({item['correct']}/{item['n']}) | "
            f"{item['delta_ex']:+.3f} | {item['gain']}/{item['loss']} | "
            f"{item['mcnemar_exact_p']:.4f} | {item['valid']:.3f} | "
            f"{item['lookup_rate']:.3f} | {item['no_answer']} |")
    lines += ["", f"**best_epoch = {best}**", ""]
    if warnings:
        lines += ["Warnings:", *[f"- {warning}" for warning in warnings], ""]
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("best_epoch", "ranking", "warnings")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
