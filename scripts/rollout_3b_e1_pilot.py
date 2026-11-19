"""20-trajectory full policy rollout under the E1 query/compact contract.

This is a PILOT, not training:
  - load the original 3B SFT policy (artifacts/checkpoints/step_600_merged)
  - build the new E1 query prompt
  - let the policy generate every action from scratch (no old action prefix)
  - execute real E1 schema/SQL tools
  - score final SQL against Gold
  - compare with the old SFT trajectory (actions, final SQL, EX)
  - record per-trajectory wall time and protocol/quality checks

Selection: 5 supervised-empty-replay cases + 5 SQL-table-mismatch cases + 10
random remaining cases. This is a stress-heavy sample, not an unbiased estimate.
"""
import argparse
import json
import random
import re
import time
import traceback
from pathlib import Path

from e1_alignment.engine import VLLMBatchEngine
from e1_alignment.runtime import Runtime, RuntimeContract, run_states
from reward.bird_ex_reward import score_sql

ACTION_TAG_RE = re.compile(r"<(schema_lookup|sql_execute|answer)>")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv_ids(path):
    import csv
    ids = []
    if not Path(path).is_file():
        return ids
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row.get("original_trajectory_id") or row.get("trajectory_id"))
    return [x for x in ids if x]


def norm_sql(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower().rstrip(";")


def action_texts(record, new=False):
    if new:
        segs = record.get("segments") or []
    else:
        segs = (record.get("trajectory") or {}).get("segments") or []
    out = []
    for s in segs:
        if s.get("kind") != "action":
            continue
        text = s.get("text") or ""
        m = ACTION_TAG_RE.search(text)
        out.append({"tag": m.group(1) if m else None, "text": text.strip()})
    return out


def normalize_action(a):
    return f"<{a['tag']}>" + re.sub(r"\s+", " ", a["text"]).strip()


def lookup_status(result):
    md = (result or {}).get("retrieval_metadata") or {}
    tables = md.get("matched_tables") if isinstance(md, dict) else None
    if tables:
        return len(tables), False, str((result or {}).get("status"))
    if (result or {}).get("tables"):
        return len((result or {}).get("tables") or []), False, str((result or {}).get("status"))
    status = str((result or {}).get("status"))
    obs = str((result or {}).get("observation") or "")
    return 0, status == "ok" and "Tables:" not in obs, status


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
    p.add_argument("--out-dir", default="reports/m6/e1_contract_rebuild/rollout20")
    p.add_argument("--n-empty", type=int, default=5)
    p.add_argument("--n-mismatch", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout20.jsonl").write_text("", encoding="utf-8")
    old_rows = list(read_jsonl(a.old_data))
    old_by_id = {r.get("trajectory_id"): r for r in old_rows}
    raw_by_qid = {}
    for r in read_jsonl(a.raw_data):
        if r.get("split") == "sft" and r.get("gold_sql"):
            raw_by_qid[r.get("id")] = r

    empty_ids = read_csv_ids(a.empty_csv)
    mismatch_ids = [x for x in read_csv_ids(a.mismatch_csv) if x not in set(empty_ids)]
    selected, used_questions = [], set()

    def add(old_id, group):
        if not old_id or old_id in {x["trajectory_id"] for x in selected}:
            return False
        old = old_by_id.get(old_id)
        if not old or old.get("final_ex") is not True:
            return False
        raw = raw_by_qid.get(old.get("question_id"))
        if not raw:
            return False
        if old.get("question_id") in used_questions:
            return False
        used_questions.add(old.get("question_id"))
        selected.append({"trajectory_id": old_id, "group": group,
                         "question_id": old.get("question_id"), "old": old, "raw": raw})
        return True

    for x in empty_ids:
        if sum(s["group"] == "empty" for s in selected) >= a.n_empty:
            break
        add(x, "empty")
    for x in mismatch_ids:
        if sum(s["group"] == "mismatch" for s in selected) >= a.n_mismatch:
            break
        add(x, "mismatch")
    rng = random.Random(a.seed)
    random_pool = [r.get("trajectory_id") for r in old_rows
                   if r.get("trajectory_id") not in {x["trajectory_id"] for x in selected}]
    rng.shuffle(random_pool)
    for x in random_pool:
        if sum(s["group"] == "random" for s in selected) >= a.n_random:
            break
        add(x, "random")
    if not selected:
        raise SystemExit("no eligible trajectories")

    contract = RuntimeContract(model_path=a.model, prompt_mode="query",
                               observation_mode="compact", max_context=a.max_model_len)
    runtime = Runtime(contract)
    print(json.dumps({"selected": len(selected),
                      "groups": {g: sum(s["group"] == g for s in selected)
                                 for g in ("empty", "mismatch", "random")},
                      "model": a.model, "temperature": a.temperature}), flush=True)

    load_started = time.monotonic()
    engine = VLLMBatchEngine(a.model, max_model_len=a.max_model_len,
                             gpu_memory_utilization=a.gpu_memory_utilization)
    load_s = round(time.monotonic() - load_started, 1)
    print(json.dumps({"engine_loaded_s": load_s}), flush=True)

    results = []
    t_all = time.monotonic()
    for index, item in enumerate(selected, 1):
        old, raw = item["old"], item["raw"]
        row = {"id": raw["id"], "question": raw["question"], "evidence": raw.get("evidence", ""),
               "gold_sql": raw["gold_sql"], "db_id": raw["db_id"], "db_path": raw["db_path"]}
        started = time.monotonic()
        err = None
        try:
            state = runtime.new_state(row)
            state.index = index - 1
            run_states([state], engine, temperature=a.temperature, top_p=1.0,
                       seed_base=a.seed + index * 1009)
            new = state.record()
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            new = {"finish_reason": "exception", "segments": [], "tool_events": [],
                   "final_sql": None, "invalid_actions": None, "infrastructure_error": True,
                   "generation_error": err, "rounds": 0, "tool_calls": 0,
                   "schema_lookups": 0, "sql_executes": 0}
        run_s = time.monotonic() - started

        old_actions = action_texts(old)
        new_actions = action_texts(new, new=True)
        old_tags = [x["tag"] for x in old_actions]
        new_tags = [x["tag"] for x in new_actions]
        old_final = old.get("final_sql")
        new_final = new.get("final_sql")

        score_started = time.monotonic()
        try:
            old_score = score_sql(raw["db_path"], old_final, raw["gold_sql"], timeout=30) if old_final else {}
            new_score = score_sql(raw["db_path"], new_final, raw["gold_sql"], timeout=30) if new_final else {}
            score_error = None
        except Exception as exc:
            old_score, new_score = {}, {}
            score_error = f"{type(exc).__name__}: {exc}"
        score_s = time.monotonic() - score_started

        lookup_empty = lookup_error = sql_error = 0
        for event in new.get("tool_events") or []:
            action = event.get("action")
            result = event.get("result") or {}
            if action == "schema_lookup":
                _, empty, status = lookup_status(result)
                lookup_empty += int(empty)
                lookup_error += int(status != "ok")
            elif action == "sql_execute":
                sql_error += int(str(result.get("status")) != "ok")

        record = {
            "index": index, "group": item["group"],
            "trajectory_id": item["trajectory_id"], "question_id": item["question_id"],
            "db_id": raw["db_id"],
            "old": {"actions": old_actions, "action_tags": old_tags,
                    "final_sql": old_final, "final_ex": bool(old_score.get("correct")),
                    "final_ex_expected": old.get("final_ex"),
                    "n_actions": len(old_actions), "n_tools": sum(t in ("schema_lookup", "sql_execute") for t in old_tags)},
            "new": {"actions": new_actions, "action_tags": new_tags,
                    "final_sql": new_final, "final_ex": bool(new_score.get("correct")),
                    "finish_reason": new.get("finish_reason"), "invalid_actions": new.get("invalid_actions"),
                    "infrastructure_error": bool(new.get("infrastructure_error")),
                    "generation_error": new.get("generation_error"),
                    "n_actions": len(new_actions), "n_tools": new.get("tool_calls"),
                    "schema_lookups": new.get("schema_lookups"), "sql_executes": new.get("sql_executes"),
                    "rounds": new.get("rounds"),
                    "lookup_empty": lookup_empty, "lookup_error": lookup_error, "sql_error": sql_error,
                    "segments": new.get("segments"), "tool_events": new.get("tool_events")},
            "compare": {
                "action_count_equal": len(old_actions) == len(new_actions),
                "tag_sequence_match": old_tags == new_tags,
                "first_tag_match": bool(old_tags and new_tags and old_tags[0] == new_tags[0]),
                "action_sequence_exact": [normalize_action(x) for x in old_actions] ==
                                         [normalize_action(x) for x in new_actions],
                "final_sql_exact": (old_final or "").strip() == (new_final or "").strip(),
                "final_sql_normalized": norm_sql(old_final) == norm_sql(new_final),
                "old_ex": bool(old_score.get("correct")), "new_ex": bool(new_score.get("correct")),
            },
            "timing": {"run_s": round(run_s, 2), "score_s": round(score_s, 2),
                       "error": err, "score_error": score_error},
        }
        results.append(record)
        with (out / "rollout20.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"i": index, "group": item["group"], "old_tags": old_tags,
                          "new_tags": new_tags, "old_ex": record["compare"]["old_ex"],
                          "new_ex": record["compare"]["new_ex"],
                          "finish": new.get("finish_reason"), "invalid": new.get("invalid_actions"),
                          "run_s": round(run_s, 1)}), flush=True)

    # fresh file for reruns
    total_s = time.monotonic() - t_all
    times = [r["timing"]["run_s"] for r in results]
    times_sorted = sorted(times)
    n = len(results)
    summary = {
        "n": n, "groups": {g: sum(r["group"] == g for r in results)
                           for g in ("empty", "mismatch", "random")},
        "engine_load_s": load_s, "total_rollout_s": round(total_s, 1),
        "time_mean_s": round(sum(times) / n, 2), "time_p50_s": round(times_sorted[n // 2], 2),
        "time_max_s": round(max(times), 2),
        "old_ex_correct": sum(r["compare"]["old_ex"] for r in results),
        "new_ex_correct": sum(r["compare"]["new_ex"] for r in results),
        "new_ex_rate": round(sum(r["compare"]["new_ex"] for r in results) / n, 3),
        "new_no_answer": sum(not r["new"]["final_sql"] for r in results),
        "new_invalid_actions": sum(int(r["new"].get("invalid_actions") or 0) for r in results),
        "new_infrastructure_errors": sum(bool(r["new"].get("infrastructure_error")) for r in results),
        "exceptions": sum(bool(r["timing"]["error"]) for r in results),
        "score_errors": sum(bool(r["timing"]["score_error"]) for r in results),
        "finish_reasons": {},
        "tag_sequence_match": sum(r["compare"]["tag_sequence_match"] for r in results),
        "first_tag_match": sum(r["compare"]["first_tag_match"] for r in results),
        "action_sequence_exact": sum(r["compare"]["action_sequence_exact"] for r in results),
        "action_count_equal": sum(r["compare"]["action_count_equal"] for r in results),
        "final_sql_exact": sum(r["compare"]["final_sql_exact"] for r in results),
        "final_sql_normalized": sum(r["compare"]["final_sql_normalized"] for r in results),
        "new_lookup_empty": sum(r["new"]["lookup_empty"] for r in results),
        "new_lookup_error": sum(r["new"]["lookup_error"] for r in results),
        "new_sql_error": sum(r["new"]["sql_error"] for r in results),
        "estimated_full_4471_s": round(sum(times) / n * 4471, 1),
        "estimated_full_4471_h": round(sum(times) / n * 4471 / 3600, 2),
        "note": ("20-trajectory stress-heavy pilot; greedy; per-trajectory run_s excludes "
                 "engine load and scoring. Full-run estimate is mean run_s * 4,471, "
                 "single-card sequential, excluding engine load and any retries."),
    }
    from collections import Counter
    summary["finish_reasons"] = dict(Counter(str(r["new"].get("finish_reason")) for r in results))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# M6-08m 20 条完整 policy rollout pilot", ""]
    lines.append(f"- 样本：{n}（empty {summary['groups']['empty']} / mismatch {summary['groups']['mismatch']} / random {summary['groups']['random']}）")
    lines.append(f"- 模型：`{a.model}`；greedy；新 E1 query/compact 契约。")
    lines.append(f"- 旧 EX：{summary['old_ex_correct']}/{n}；新 EX：**{summary['new_ex_correct']}/{n}**（{summary['new_ex_rate']:.1%}）。")
    lines.append(f"- 新无 answer：{summary['new_no_answer']}；invalid_actions：{summary['new_invalid_actions']}；"
                 f"infrastructure：{summary['new_infrastructure_errors']}；exceptions：{summary['exceptions']}。")
    lines.append(f"- 轨迹变化：tag 序列完全一致 {summary['tag_sequence_match']}/{n}；首 action tag 一致 "
                 f"{summary['first_tag_match']}/{n}；action 序列完全一致 {summary['action_sequence_exact']}/{n}；"
                 f"final SQL exact {summary['final_sql_exact']}/{n} / normalized {summary['final_sql_normalized']}/{n}。")
    lines.append(f"- 新轨迹 lookup empty {summary['new_lookup_empty']}；lookup error {summary['new_lookup_error']}；"
                 f"SQL error {summary['new_sql_error']}；finish：{summary['finish_reasons']}。")
    lines.append(f"- 单条时间：mean **{summary['time_mean_s']:.1f}s** / p50 {summary['time_p50_s']:.1f}s / max {summary['time_max_s']:.1f}s；"
                 f"engine load {summary['engine_load_s']:.1f}s；pilot total {summary['total_rollout_s']:.1f}s。")
    lines.append(f"- 按 mean × 4,471 外推：**{summary['estimated_full_4471_h']:.1f} h**（单卡串行，不含 engine load/重试；外推）。")
    lines.append("")
    lines.append("## 质量/差异摘要")
    lines.append("- `tag_sequence_match` 低说明新 observation 下策略选择的 action 序列与旧数据不同；"
                 "`final_sql_normalized` 低说明旧 answer 不能直接复用。")
    lines.append("- 若 `new_no_answer` / `invalid_actions` > 0，说明新 prompt 下 3B 策略的停止/格式仍需处理。")
    lines.append("- 本 pilot 是 stress-heavy 样本（5 empty + 5 mismatch + 10 random），不是无偏时间估计。")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
