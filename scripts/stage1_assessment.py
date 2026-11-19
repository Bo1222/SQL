"""Stage 1 assessment: three layers (engineering / learning signal / task ability).

Consumes the training log, the dev200 capability files and the group-signal report,
then classifies the run into scenario A (task gain), B (policy moves, no task gain)
or C (drift/collapse).
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path


def load(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_steps(path):
    path = Path(path)
    if not path.is_file():
        return []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pair = re.compile(r"([A-Za-z0-9_/\.]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ansi.sub("", line)
        if "step:" not in line or "global_seqlen/min" not in line:
            continue
        kv = {k: float(v) for k, v in pair.findall(line)}
        if "training/global_step" in kv:
            rows.append({k: kv.get(k) for k in
                         ("training/global_step", "critic/score/mean", "actor/kl_loss", "actor/entropy",
                          "actor/grad_norm", "actor/pg_loss", "timing_s/step", "response_length/mean")})
    return rows


def slope(values):
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = st.mean(xs), st.mean(values)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom if denom else 0.0


def mcnemar(gain, loss):
    import math
    d = gain + loss
    if not d:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / 2 ** d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/m6_stage1.log")
    p.add_argument("--baseline", default="reports/m6/capability_sft.json")
    p.add_argument("--rl", nargs="+", default=["reports/m6/capability_rl25.json", "reports/m6/capability_rl50.json"])
    p.add_argument("--group-signal", default="reports/m6/group_signal_stage1.json")
    p.add_argument("--out", default="reports/m6/stage1_assessment.json")
    a = p.parse_args()

    steps = parse_steps(a.log)
    warm = [s for s in steps if s["training/global_step"] > 1] or steps
    engineering = {
        "steps_logged": len(steps),
        "kl_max": round(max(s["actor/kl_loss"] for s in warm), 6) if warm else None,
        "kl_slope_per_step": round(slope([s["actor/kl_loss"] for s in warm]), 8) if warm else None,
        "entropy_min": round(min(s["actor/entropy"] for s in warm), 4) if warm else None,
        "entropy_slope_per_step": round(slope([s["actor/entropy"] for s in warm]), 6) if warm else None,
        "grad_norm_min": round(min(s["actor/grad_norm"] for s in warm), 4) if warm else None,
        "grad_norm_max": round(max(s["actor/grad_norm"] for s in warm), 4) if warm else None,
        "step_time_mean_s": round(st.mean(s["timing_s/step"] for s in warm), 1) if warm else None,
        "response_tokens_mean": round(st.mean(s["response_length/mean"] for s in warm), 1) if warm else None,
    }
    learning = {
        "reward_mean": round(st.mean(s["critic/score/mean"] for s in warm), 4) if warm else None,
        "reward_slope_per_step": round(slope([s["critic/score/mean"] for s in warm]), 6) if warm else None,
        "pg_loss_mean": round(st.mean(s["actor/pg_loss"] for s in warm), 6) if warm and warm[0].get("actor/pg_loss") is not None else None,
    }
    groups = load(a.group_signal) or {}
    if groups.get("summary"):
        learning.update({"mixed_rate": groups["summary"].get("mixed_rate"),
                         "all_zero": groups["summary"].get("all_zero"),
                         "all_correct": groups["summary"].get("all_correct"),
                         "effective_pg_groups_per_step": groups["summary"].get("effective_pg_groups_per_step")})

    baseline = (load(a.baseline) or {}).get("summary", {})
    task = {"sft": baseline}
    for path in a.rl:
        data = load(path)
        if not data:
            continue
        tag = Path(path).stem.replace("capability_", "")
        task[tag] = data["summary"]
    # paired comparison vs SFT for the last RL checkpoint available
    per_question = {}
    base_records = Path("reports/m6/dev200_baseline.jsonl")
    if base_records.is_file():
        per_question["sft"] = {r["id"]: bool(r.get("correct")) for r in
                               (json.loads(l) for l in base_records.read_text(encoding="utf-8").splitlines() if l.strip())}
    paired = {}
    for path in a.rl:
        records = Path(str(path).replace("capability_", "dev200_").replace(".json", ".jsonl"))
        if not records.is_file() or not per_question.get("sft"):
            continue
        rl = {r["id"]: bool(r.get("correct")) for r in
              (json.loads(l) for l in records.read_text(encoding="utf-8").splitlines() if l.strip())}
        common = sorted(set(per_question["sft"]) & set(rl))
        gain = [i for i in common if rl[i] and not per_question["sft"][i]]
        loss = [i for i in common if not rl[i] and per_question["sft"][i]]
        paired[Path(path).stem.replace("capability_", "")] = {
            "n": len(common), "gain": len(gain), "loss": len(loss), "net": len(gain) - len(loss),
            "mcnemar_p": round(mcnemar(len(gain), len(loss)), 4),
            "delta_ex": round((len(gain) - len(loss)) / max(len(common), 1), 4),
        }
    task["paired_vs_sft"] = paired

    # scenario classification (uses the latest checkpoint with a dev200 evaluation)
    latest = task.get("rl50") or task.get("rl25")
    scenario, reason = "INSUFFICIENT", "dev200 evaluation missing"
    if latest and baseline:
        d_ex = latest["final_ex"] - baseline["final_ex"]
        d_first = latest["first_sql_ex"] - baseline["first_sql_ex"]
        d_repair = (latest.get("repair_rate") or 0) - (baseline.get("repair_rate") or 0)
        kl_up = (engineering.get("kl_slope_per_step") or 0) > 0
        ent_down = (engineering.get("entropy_slope_per_step") or 0) < 0
        if d_ex > 0 and (d_first > 0 or d_repair > 0):
            scenario, reason = "A", "final EX up together with first-SQL or repair improvement"
        elif d_ex <= 0 and (abs(d_first) > 0.005 or abs(d_repair) > 0.01):
            scenario, reason = "B", "policy/behavior moved but final EX did not improve"
        elif kl_up and ent_down and d_ex < 0:
            scenario, reason = "C", "KL rising + entropy falling + final EX down"
        else:
            scenario, reason = "B", "no clear task-level gain"
        task["deltas"] = {"final_ex": round(d_ex, 4), "first_sql_ex": round(d_first, 4),
                          "repair_rate": round(d_repair, 4)}
    report = {"format": "m6-stage1-assessment-v1", "engineering": engineering,
              "learning_signal": learning, "task_ability": task,
              "scenario": scenario, "scenario_reason": reason}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2200])


if __name__ == "__main__":
    main()
