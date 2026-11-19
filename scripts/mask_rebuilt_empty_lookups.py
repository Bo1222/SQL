"""Create a conservative cleaned candidate from the rebuilt E1-contract SFT data.

Rule: if a supervised schema_lookup returned no matched tables under E1
(`retrieval_metadata.matched_tables` empty and no "Tables:" in observation),
mask that action's supervision. Keep the action and the empty observation in
context; do not delete the trajectory. Recompute trajectory.loss_mask and
masked_tokens.

This does NOT touch the raw rebuilt file and does NOT train.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def lookup_tables(result):
    md = (result or {}).get("retrieval_metadata") or {}
    if isinstance(md, dict) and md.get("matched_tables"):
        return [str(x) for x in md["matched_tables"]]
    tables = (result or {}).get("tables")
    if isinstance(tables, list):
        return [str(t.get("name")) for t in tables if isinstance(t, dict) and t.get("name")]
    return []


def is_empty_lookup(result):
    if not isinstance(result, dict):
        return False
    if str(result.get("status")) != "ok":
        return False
    if lookup_tables(result):
        return False
    return "Tables:" not in str(result.get("observation") or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/m6/e1_contract_rebuild/sft_trajectories_e1.jsonl")
    p.add_argument("--out", default="artifacts/m6/e1_contract_rebuild/sft_trajectories_e1_clean.jsonl")
    p.add_argument("--manifest", default="artifacts/m6/e1_contract_rebuild/clean_manifest.json")
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    before_sup = after_sup = 0
    before_masked = after_masked = 0
    with Path(a.out).open("w", encoding="utf-8") as f:
        for rec in read_jsonl(a.data):
            counts["records"] += 1
            traj = rec.get("trajectory") or {}
            segs = traj.get("segments") or []
            events = traj.get("tool_events") or []
            by_event = {int(e.get("tool_event_index", i)): e for i, e in enumerate(events)}
            masked_here = 0
            for seg in segs:
                if seg.get("kind") != "action" or seg.get("action") != "schema_lookup":
                    continue
                event = by_event.get(int(seg.get("tool_event_index", -1)))
                if is_empty_lookup((event or {}).get("result")):
                    counts["empty_lookups"] += 1
                    if seg.get("supervise") is True:
                        seg["supervise"] = False
                        seg["masked_reason"] = "e1_empty_lookup"
                        masked_here += 1
                        counts["masked_supervised_empty"] += 1
            loss_mask = []
            for seg in segs:
                sup = seg.get("kind") == "action" and seg.get("supervise") is True
                loss_mask.extend([1 if sup else 0] * len(seg["ids"]))
            traj["loss_mask"] = loss_mask
            rec["trajectory"] = traj
            rec["masked_tokens"] = sum(loss_mask)
            rec["clean"] = {"masked_supervised_empty_lookups": masked_here,
                            "policy": "mask_e1_empty_lookup_supervision"}
            before_sup += sum(1 for s in segs if s.get("kind") == "action" and s.get("supervise") is True) + masked_here
            after_sup += sum(1 for s in segs if s.get("kind") == "action" and s.get("supervise") is True)
            before_masked += int(rec.get("masked_tokens") or 0) + sum(
                len(seg["ids"]) for seg in segs
                if seg.get("kind") == "action" and seg.get("masked_reason") == "e1_empty_lookup")
            after_masked += int(rec.get("masked_tokens") or 0)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {"data": a.data, "out": a.out,
                "records": counts["records"], "empty_lookups": counts["empty_lookups"],
                "masked_supervised_empty": counts["masked_supervised_empty"],
                "supervised_actions_before": before_sup,
                "supervised_actions_after": after_sup,
                "masked_tokens_before": before_masked, "masked_tokens_after": after_masked,
                "note": "Raw rebuilt file is unchanged; this is a conservative candidate that masks "
                        "supervised empty lookups. It does not repair behavior or train."}
    Path(a.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
