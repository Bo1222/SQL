"""Contract ablation: old/new prompt x old/new observation on the same 20 questions.

Conditions (same model, same questions, greedy, same batch size):
  query_compact  = new E1 query prompt + new compact observation (current)
  sft_compact    = old SFT table prompt + new compact observation
  query_legacy   = new query prompt + legacy full-payload observation
  sft_legacy     = old SFT prompt + legacy observation (closest to old training contract)

This isolates whether the large old->new difference comes from the prompt
contract, the observation format, or the policy/task itself.
"""
import argparse
import json
import random
import re
import time
from collections import Counter
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


def obs_truncated(segments):
    return any(s.get("kind") == "observation" and (
        '"truncated": true' in (s.get("text") or "") or "'truncated': True" in (s.get("text") or "")
    ) for s in (segments or []))


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
    p.add_argument("--out-dir", default="reports/m6/e1_contract_rebuild/contract_ablation20")
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

    contract_args = dict(model_path=a.model, max_context=a.max_model_len)
    engine = VLLMBatchEngine(a.model, max_model_len=a.max_model_len,
                             gpu_memory_utilization=a.gpu_memory_utilization)
    conditions = [
        ("query_compact", "query", "compact"),
        ("sft_compact", "sft", "compact"),
        ("query_legacy", "query", "legacy"),
        ("sft_legacy", "sft", "legacy"),
    ]
    all_results = []
    summary = {"n": len(selected), "groups": dict(Counter(s["group"] for s in selected)),
               "old_ex": 0, "conditions": {}, "note": "same 20 questions, greedy, batch-20; old EX scored from old final SQL"}
    for item in selected:
        raw = item["raw"]
        summary["old_ex"] += int(score_sql(raw["db_path"], item["old"].get("final_sql"), raw["gold_sql"], timeout=30).get("correct", False))

    for name, prompt_mode, observation_mode in conditions:
        contract = RuntimeContract(**contract_args, prompt_mode=prompt_mode,
                                   observation_mode=observation_mode)
        runtime = Runtime(contract)
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
        elapsed = round(time.monotonic() - started, 1)
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
            old_tags = action_tags((item["old"].get("trajectory") or {}).get("segments"))
            look_empty = sum(lookup_empty((e.get("result") or {})) for e in (new.get("tool_events") or [])
                             if e.get("action") == "schema_lookup")
            sql_err = sum(str((e.get("result") or {}).get("status")) != "ok" for e in (new.get("tool_events") or [])
                          if e.get("action") == "sql_execute")
            rows.append({"group": item["group"], "trajectory_id": item["trajectory_id"],
                         "old_tags": old_tags, "new_tags": tags,
                         "new_final": new.get("final_sql"), "new_ex": bool(scored.get("correct")),
                         "finish_reason": new.get("finish_reason"),
                         "invalid_actions": new.get("invalid_actions"),
                         "infrastructure_error": new.get("infrastructure_error"),
                         "score_error": score_err,
                         "tool_calls": new.get("tool_calls"), "rounds": new.get("rounds"),
                         "lookup_empty": look_empty, "sql_error": sql_err,
                         "obs_truncated": obs_truncated(new.get("segments") or []),
                         "tag_sequence_match": old_tags == tags})
        groups = {}
        for g in ("empty", "mismatch", "random"):
            sub = [r for r in rows if r["group"] == g]
            groups[g] = {"n": len(sub), "new_ex": sum(r["new_ex"] for r in sub),
                         "tag_match": sum(r["tag_sequence_match"] for r in sub)}
        summ = {
            "prompt_mode": prompt_mode, "observation_mode": observation_mode,
            "batch_wall_s": elapsed,
            "new_ex": sum(r["new_ex"] for r in rows),
            "new_ex_rate": round(sum(r["new_ex"] for r in rows) / max(len(rows), 1), 3),
            "groups": groups,
            "finish_reasons": dict(Counter(str(r["finish_reason"]) for r in rows)),
            "invalid_actions": sum(int(r["invalid_actions"] or 0) for r in rows),
            "infrastructure_errors": sum(bool(r["infrastructure_error"]) for r in rows),
            "score_errors": sum(bool(r["score_error"]) for r in rows),
            "no_answer": sum(not r["new_final"] for r in rows),
            "tag_sequence_match": sum(r["tag_sequence_match"] for r in rows),
            "lookup_empty": sum(r["lookup_empty"] for r in rows),
            "sql_error": sum(r["sql_error"] for r in rows),
            "obs_truncated_trajectories": sum(r["obs_truncated"] for r in rows),
        }
        summary["conditions"][name] = summ
        all_results.append({"condition": name, "rows": rows})
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(json.dumps({"condition": name, **summ}, ensure_ascii=False), flush=True)

    (out / "ablation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 20 条契约 ablation：old/new prompt × old/new observation", "",
             f"- 样本：{summary['n']}（{summary['groups']}）；模型 `{a.model}`；greedy；batch-20；旧 EX {summary['old_ex']}/{summary['n']}。", "",
             "| condition | prompt | observation | new EX | tag match | lookup empty | SQL error | invalid | infra | no answer | wall s |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"query_compact": ("new query", "new compact"), "sft_compact": ("old SFT", "new compact"),
              "query_legacy": ("new query", "legacy full"), "sft_legacy": ("old SFT", "legacy full")}
    for name, _, _ in conditions:
        s = summary["conditions"][name]
        lines.append(f"| {name} | {labels[name][0]} | {labels[name][1]} | {s['new_ex']}/{summary['n']} | "
                     f"{s['tag_sequence_match']}/{summary['n']} | {s['lookup_empty']} | {s['sql_error']} | "
                     f"{s['invalid_actions']} | {s['infrastructure_errors']} | {s['no_answer']} | {s['batch_wall_s']:.1f} |")
    lines += ["", "> 这是 20 条 stress-heavy 诊断，不是 dev200 估计；batch 生成可能有非确定性。"]
    (out / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
