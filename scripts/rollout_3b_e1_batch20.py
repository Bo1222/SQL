"""Batch 20-trajectory throughput test for the E1 policy rollout.

Same selection as the sequential pilot (5 empty + 5 mismatch + 10 random),
but all 20 states are advanced in lockstep with run_states.  This measures
wall-clock throughput, not per-trajectory latency.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

from e1_alignment.engine import VLLMBatchEngine
from e1_alignment.runtime import Runtime, RuntimeContract, run_states
from reward.bird_ex_reward import score_sql

TAG_RE = re.compile(r"<(schema_lookup|sql_execute|answer)>")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv_ids(path):
    import csv
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [r.get("original_trajectory_id") or r.get("trajectory_id")
                for r in csv.DictReader(f) if r.get("original_trajectory_id") or r.get("trajectory_id")]


def action_tags(segments):
    tags = []
    for s in segments or []:
        if s.get("kind") != "action":
            continue
        m = TAG_RE.search(s.get("text") or "")
        tags.append(m.group(1) if m else None)
    return tags


def lookup_empty(result):
    md = (result or {}).get("retrieval_metadata") or {}
    if isinstance(md, dict) and md.get("matched_tables"):
        return False
    if (result or {}).get("tables"):
        return False
    return str((result or {}).get("status")) == "ok" and "Tables:" not in str((result or {}).get("observation") or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-data", default="artifacts/datasets/sft_trajectories.jsonl")
    p.add_argument("--raw-data",
                   default="data/processed/bird_agent/full_sft_20261026/full_tool_unique.jsonl")
    p.add_argument("--model", default="artifacts/checkpoints/step_600_merged")
    p.add_argument("--empty-csv",
                   default="reports/m6/e1_contract_rebuild/audit/lookup_supervised_empty.csv")
    p.add_argument("--mismatch-csv",
                   default="reports/m6/e1_contract_rebuild/audit/lookup_sql_table_mismatch.csv")
    p.add_argument("--out-dir", default="reports/m6/e1_contract_rebuild/rollout20_batch")
    p.add_argument("--n-empty", type=int, default=5)
    p.add_argument("--n-mismatch", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    old_rows = list(read_jsonl(a.old_data))
    old_by_id = {r.get("trajectory_id"): r for r in old_rows}
    raw_by_qid = {r.get("id"): r for r in read_jsonl(a.raw_data)
                  if r.get("split") == "sft" and r.get("gold_sql")}
    empty_ids = read_csv_ids(a.empty_csv)
    mismatch_ids = [x for x in read_csv_ids(a.mismatch_csv) if x not in set(empty_ids)]
    selected, used_q = [], set()

    def add(old_id, group):
        if old_id in {x["trajectory_id"] for x in selected}:
            return
        old = old_by_id.get(old_id)
        if not old or old.get("final_ex") is not True:
            return
        raw = raw_by_qid.get(old.get("question_id"))
        if not raw or old.get("question_id") in used_q:
            return
        used_q.add(old.get("question_id"))
        selected.append({"trajectory_id": old_id, "group": group, "old": old, "raw": raw})

    for x in empty_ids:
        if sum(s["group"] == "empty" for s in selected) >= a.n_empty:
            break
        add(x, "empty")
    for x in mismatch_ids:
        if sum(s["group"] == "mismatch" for s in selected) >= a.n_mismatch:
            break
        add(x, "mismatch")
    rng = random.Random(a.seed)
    pool = [r.get("trajectory_id") for r in old_rows
            if r.get("trajectory_id") not in {x["trajectory_id"] for x in selected}]
    rng.shuffle(pool)
    for x in pool:
        if sum(s["group"] == "random" for s in selected) >= a.n_random:
            break
        add(x, "random")
    if len(selected) != a.n_empty + a.n_mismatch + a.n_random:
        print(json.dumps({"warning": "selection shorter than requested", "selected": len(selected)}), flush=True)

    contract = RuntimeContract(model_path=a.model, prompt_mode="query",
                               observation_mode="compact", max_context=a.max_model_len)
    runtime = Runtime(contract)
    load_t = time.monotonic()
    engine = VLLMBatchEngine(a.model, max_model_len=a.max_model_len,
                             gpu_memory_utilization=a.gpu_memory_utilization)
    load_s = time.monotonic() - load_t

    states = []
    for i, item in enumerate(selected):
        raw = item["raw"]
        row = {"id": raw["id"], "question": raw["question"], "evidence": raw.get("evidence", ""),
               "gold_sql": raw["gold_sql"], "db_id": raw["db_id"], "db_path": raw["db_path"]}
        state = runtime.new_state(row)
        state.index = i
        states.append(state)
    started = time.monotonic()
    run_states(states, engine, temperature=0.0, top_p=1.0, seed_base=a.seed)
    batch_s = time.monotonic() - started

    rows = []
    for item, state in zip(selected, states):
        raw = item["raw"]
        new = state.record()
        try:
            scored = score_sql(raw["db_path"], new.get("final_sql"), raw["gold_sql"], timeout=30) if new.get("final_sql") else {}
            score_err = None
        except Exception as exc:
            scored, score_err = {}, f"{type(exc).__name__}: {exc}"
        tags = action_tags(new.get("segments"))
        look_empty = sum(lookup_empty((e.get("result") or {})) for e in (new.get("tool_events") or [])
                         if e.get("action") == "schema_lookup")
        sql_err = sum(str((e.get("result") or {}).get("status")) != "ok" for e in (new.get("tool_events") or [])
                      if e.get("action") == "sql_execute")
        old_tags = action_tags((item["old"].get("trajectory") or {}).get("segments"))
        rows.append({"group": item["group"], "trajectory_id": item["trajectory_id"],
                     "old_tags": old_tags, "new_tags": tags,
                     "old_final": item["old"].get("final_sql"), "new_final": new.get("final_sql"),
                     "old_ex": score_sql(raw["db_path"], item["old"].get("final_sql"), raw["gold_sql"], timeout=30).get("correct", False),
                     "new_ex": bool(scored.get("correct")), "finish_reason": new.get("finish_reason"),
                     "invalid_actions": new.get("invalid_actions"),
                     "infrastructure_error": new.get("infrastructure_error"),
                     "score_error": score_err,
                     "tool_calls": new.get("tool_calls"), "rounds": new.get("rounds"),
                     "lookup_empty": look_empty, "sql_error": sql_err,
                     "tag_sequence_match": old_tags == tags})
    times = batch_s / len(rows) if rows else None
    groups = {}
    for g in ("empty", "mismatch", "random"):
        sub = [r for r in rows if r["group"] == g]
        groups[g] = {"n": len(sub), "old_ex": sum(r["old_ex"] for r in sub),
                     "new_ex": sum(r["new_ex"] for r in sub),
                     "tag_match": sum(r["tag_sequence_match"] for r in sub)}
    summary = {
        "n": len(rows), "groups": groups,
        "engine_load_s": round(load_s, 1), "batch_wall_s": round(batch_s, 1),
        "amortized_s_per_trajectory": round(times, 2) if times else None,
        "old_ex": sum(r["old_ex"] for r in rows), "new_ex": sum(r["new_ex"] for r in rows),
        "new_ex_rate": round(sum(r["new_ex"] for r in rows) / max(len(rows), 1), 3),
        "finish_reasons": {}, "invalid_actions": sum(int(r["invalid_actions"] or 0) for r in rows),
        "infrastructure_errors": sum(bool(r["infrastructure_error"]) for r in rows),
        "score_errors": sum(bool(r["score_error"]) for r in rows),
        "no_answer": sum(not r["new_final"] for r in rows),
        "tag_sequence_match": sum(r["tag_sequence_match"] for r in rows),
        "lookup_empty": sum(r["lookup_empty"] for r in rows),
        "sql_error": sum(r["sql_error"] for r in rows),
        "estimated_full_4471_s": round(times * 4471, 1) if times else None,
        "estimated_full_4471_h": round(times * 4471 / 3600, 2) if times else None,
        "note": ("Batch-20 throughput test; amortized_s_per_trajectory = batch_wall_s / 20, "
                 "not per-trajectory latency. Sequential pilot measured 12.43 s/trajectory. "
                 "Full estimate assumes the same batch size and no retries."),
    }
    from collections import Counter
    summary["finish_reasons"] = dict(Counter(str(r["finish_reason"]) for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    lines = ["# Batch-20 E1 policy rollout throughput", ""]
    lines.append(f"- n={len(rows)}；groups={groups}")
    lines.append(f"- old EX {summary['old_ex']}/{len(rows)}；new EX {summary['new_ex']}/{len(rows)}（{summary['new_ex_rate']:.1%}）")
    lines.append(f"- tags match {summary['tag_sequence_match']}/{len(rows)}；lookup empty {summary['lookup_empty']}；SQL error {summary['sql_error']}")
    lines.append(f"- engine load {summary['engine_load_s']:.1f}s；batch wall **{summary['batch_wall_s']:.1f}s**；"
                 f"amortized **{summary['amortized_s_per_trajectory']:.2f}s/条**")
    lines.append(f"- 按 batch/20 × 4,471 外推：**{summary['estimated_full_4471_h']:.2f} h**（同 batch size，不含重试；外推）")
    lines.append(f"- finish：{summary['finish_reasons']}；invalid {summary['invalid_actions']}；"
                 f"infra {summary['infrastructure_errors']}；no answer {summary['no_answer']}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
