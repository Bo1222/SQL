"""Export the 20 old/new trajectories as a single review-only Markdown file.

No analysis, no recommendations: only prompts, actions, observations, final SQL,
EX and timing.  The file is intended for manual audit.
"""
import json
from pathlib import Path

OLD = Path("artifacts/datasets/sft_trajectories.jsonl")
RAW = Path("data/processed/bird_agent/full_sft_20261026/full_tool_unique.jsonl")
ROLL = Path("reports/m6/e1_contract_rebuild/rollout20/rollout20.jsonl")
OUT = Path("reports/m6/e1_contract_rebuild/rollout20/review_trajectories.md")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def segment_text(segments):
    out = []
    for seg in segments or []:
        kind = seg.get("kind", "?")
        text = seg.get("text") or ""
        if kind == "prompt":
            out.append(("prompt", text))
        elif kind == "action":
            action = seg.get("action") or "?"
            out.append(("action", f"[{action}]\n{text}"))
        elif kind == "observation":
            out.append(("observation", text))
        else:
            out.append((kind, text))
    return out


def fence(title, text):
    return f"**{title}**\n\n~~~text\n{text.rstrip()}\n~~~\n"


def main():
    old_by_id = {r["trajectory_id"]: r for r in read_jsonl(OLD)}
    raw_by_qid = {r["id"]: r for r in read_jsonl(RAW) if r.get("split") == "sft"}
    rows = list(read_jsonl(ROLL))
    lines = [
        "# M6-08m 20 条 policy rollout 新旧轨迹审核文件",
        "",
        "> 本文件只导出内容与结果，不含分析。",
        "> 每条含：题目/evidence/gold、旧 prompt、旧 action/observation、旧 final SQL/EX；",
        "> 新 prompt、新 action/observation、新 final SQL/EX；以及新旧运行结果。",
        "> 旧轨迹来自 `artifacts/datasets/sft_trajectories.jsonl`；新轨迹来自 `rollout20.jsonl`。",
        "",
        "## 索引",
        "",
        "| # | group | trajectory_id | db_id | old EX | new EX | old actions | new actions | new run s |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['index']} | {r['group']} | `{r['trajectory_id']}` | {r['db_id']} | "
                     f"{int(bool(r['old']['final_ex']))} | {int(bool(r['new']['final_ex']))} | "
                     f"{r['old']['n_actions']} | {r['new']['n_actions']} | {r['timing']['run_s']:.2f} |")
    lines.append("")

    for r in rows:
        old = old_by_id[r["trajectory_id"]]
        raw = raw_by_qid.get(r["question_id"], {})
        lines += [
            f"## {r['index']:02d} · {r['group']} · `{r['trajectory_id']}`",
            "",
            f"- db_id: `{r['db_id']}`",
            f"- old EX: **{int(bool(r['old']['final_ex']))}** (expected `{r['old'].get('final_ex_expected')}`)",
            f"- new EX: **{int(bool(r['new']['final_ex']))}**; finish: `{r['new'].get('finish_reason')}`; "
            f"run: **{r['timing']['run_s']:.2f} s**",
            f"- old/new tag sequence: `{r['old']['action_tags']}` -> `{r['new']['action_tags']}`",
            "",
            "### Question",
            "",
            "~~~text",
            (raw.get("question") or "").rstrip(),
            "~~~",
            "",
            "### Evidence",
            "",
            "~~~text",
            (raw.get("evidence") or "(not supplied)").rstrip(),
            "~~~",
            "",
            "### Gold SQL",
            "",
            "~~~sql",
            (raw.get("gold_sql") or "").rstrip(),
            "~~~",
            "",
            "## OLD trajectory",
            "",
            "### OLD final result",
            "",
            "~~~sql",
            (r["old"].get("final_sql") or "").rstrip(),
            "~~~",
            "",
            f"OLD EX = **{int(bool(r['old']['final_ex']))}**",
            "",
            "### OLD content",
            "",
        ]
        for kind, text in segment_text((old.get("trajectory") or {}).get("segments")):
            lines.append(fence(kind.upper(), text))
        lines += [
            "## NEW trajectory",
            "",
            "### NEW final result",
            "",
            "~~~sql",
            (r["new"].get("final_sql") or "").rstrip(),
            "~~~",
            "",
            f"NEW EX = **{int(bool(r['new']['final_ex']))}**; finish = `{r['new'].get('finish_reason')}`; "
            f"run = **{r['timing']['run_s']:.2f} s**",
            "",
            "### NEW content",
            "",
        ]
        for kind, text in segment_text(r["new"].get("segments")):
            lines.append(fence(kind.upper(), text))
        lines += ["---", ""]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "bytes": OUT.stat().st_size, "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
