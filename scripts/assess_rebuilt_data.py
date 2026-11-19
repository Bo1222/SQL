"""Summarize the rebuilt E1-contract 3B SFT data and compare it with the old data."""
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


def stats(vals):
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    n = len(vals)
    return {"n": n, "mean": round(sum(vals) / n, 1), "p50": vals[n // 2],
            "p90": vals[int(n * 0.9)], "p99": vals[int(n * 0.99)], "max": max(vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/m6/e1_contract_rebuild/sft_trajectories_e1.jsonl")
    p.add_argument("--manifest", default="artifacts/m6/e1_contract_rebuild/manifest.json")
    p.add_argument("--old-data", default="artifacts/datasets/sft_trajectories.jsonl")
    p.add_argument("--out", default="reports/m6/e1_contract_rebuild/assessment.md")
    a = p.parse_args()
    rows = list(read_jsonl(a.data))
    old_rows = list(read_jsonl(a.old_data))
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8")) if Path(a.manifest).is_file() else {}
    action_counts = Counter()
    source_counts = Counter()
    supervised = 0
    tokens = []
    masked = []
    final_ex = Counter()
    prompt_contract = Counter()
    obs_contract = Counter()
    lookup_empty = 0
    sql_error = 0
    for r in rows:
        source_counts[str(r.get("source_type"))] += 1
        final_ex[str(r.get("final_ex"))] += 1
        traj = r.get("trajectory") or {}
        actions = traj.get("actions") or []
        action_counts.update(str(x) for x in actions)
        for s in traj.get("segments") or []:
            if s.get("kind") == "prompt":
                text = s.get("text") or ""
                prompt_contract["query_prompt" if '{"query"' in text else "other"] += 1
            if s.get("kind") == "action" and s.get("supervise") is True:
                supervised += 1
            if s.get("kind") == "observation":
                text = s.get("text") or ""
                if '"observation"' in text:
                    obs_contract["has_observation_key"] += 1
                if '"tables"' in text:
                    obs_contract["has_old_tables_key"] += 1
                if '"columns"' in text:
                    obs_contract["has_columns_key"] += 1
        tokens.append(int(r.get("token_count") or 0))
        masked.append(int(r.get("masked_tokens") or 0))
        rb = r.get("rebuild") or {}
        lookup_empty += int(rb.get("lookup_empty") or 0)
        sql_error += int(rb.get("sql_error") or 0)

    old_actions = 0
    old_lookups = 0
    for r in old_rows:
        segs = (r.get("trajectory") or {}).get("segments") or []
        old_actions += sum(1 for s in segs if s.get("kind") == "action")
        old_lookups += sum(1 for s in segs if s.get("kind") == "action" and
                           (s.get("text") or "").strip().startswith("<schema_lookup>"))

    lines = ["# M6-08m 3B E1 新契约轨迹重建 — 数据评估", ""]
    lines.append("> 本文件只评估重建数据；未训练、未评测任务能力。")
    lines.append("")
    lines.append("## 计数")
    lines.append(f"- 旧数据：{len(old_rows)} 条；action {old_actions}；schema_lookup {old_lookups}")
    lines.append(f"- 新数据：**{len(rows)}** 条；action {sum(action_counts.values())}；"
                 f"supervised action {supervised}")
    lines.append(f"- action 类型：{dict(action_counts)}")
    lines.append(f"- source_type：{dict(source_counts)}")
    lines.append(f"- final_ex：{dict(final_ex)}")
    lines.append(f"- prompt 契约：{dict(prompt_contract)}")
    lines.append(f"- observation 契约：{dict(obs_contract)}")
    lines.append(f"- token：{stats(tokens)}")
    lines.append(f"- supervised token：{stats(masked)}")
    lines.append(f"- per-record lookup_empty 计数：{lookup_empty}；sql_error：{sql_error}")
    if manifest:
        lines.append(f"- rebuild manifest：{json.dumps(manifest.get('counts', {}), ensure_ascii=False)}")
        lines.append(f"- skip reasons：{json.dumps(manifest.get('skip_reasons', {}), ensure_ascii=False)}")
    lines.append("")
    lines.append("## 解释")
    lines.append("- 新数据应全部为 query prompt + compact observation；如果 `has_old_tables_key` > 0，"
                 "说明仍有旧 observation 混入，需要排查。")
    lines.append("- `final_ex` 来自旧轨迹的 final_ex=1 过滤；没有重新用 Gold 评分。"
                 "最终 EX 未变的前提是 final_sql 和 DB/Gold 未变。")
    lines.append("- 本轮只重建数据，不训练；下一步是重建后的 lookup 审计（41/368 对比）。")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "actions": sum(action_counts.values()),
                      "supervised_actions": supervised, "source": dict(source_counts),
                      "final_ex": dict(final_ex), "prompt_contract": dict(prompt_contract),
                      "obs_contract": dict(obs_contract), "token": stats(tokens),
                      "supervised_token": stats(masked)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
