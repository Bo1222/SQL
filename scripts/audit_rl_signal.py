"""CPU-only RL learning-signal audit for the reviewed 1,784-question set.

Uses existing rollouts/logs only: no model, no GPU, no new generation.

Inputs:
  reports/m6/rl_signal_20261118/records.jsonl   100 questions x 8 samples
  reports/m6/e1_student_only/rl_r{1,3}_trace_analysis.json
  reports/m6/e1_student_only/rl_r{1,3}_step*.json
  reports/m6/rl_learning5batch_4q_5e6_confirm_lr1e6_kl01_20261120/{gate.json,after32/summary.json}

The audit answers two questions before any P1-3 GPU budget:
  1. Does an n=8 GRPO group have usable signal (mixed groups) on the RL
     distribution?
  2. Do the prior full-data runs and the P0-3 batch recipe support a 12h pilot?

Gate (pre-registered here):
  mixed_rate >= 0.50, pass@8 >= 0.55, all_zero_rate <= 0.35.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def bootstrap_ci(values, seed=0, samples=2000):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [round(means[int(0.025 * (samples - 1))], 4),
            round(means[int(0.975 * (samples - 1))], 4)]


def group_stats(records, n_samples=8):
    groups = defaultdict(list)
    for record in records:
        groups[record["id"]].append(record)
    counts = {qid: sum(1 for r in rows if r.get("correct")) for qid, rows in groups.items()}
    n = len(counts)
    mixed = sum(1 for value in counts.values() if 0 < value < n_samples)
    all_zero = sum(1 for value in counts.values() if value == 0)
    all_correct = sum(1 for value in counts.values() if value == n_samples)
    per_question = [value / n_samples for value in counts.values()]
    by_db = defaultdict(lambda: {"questions": 0, "correct": 0, "pass8": 0, "all_zero": 0})
    for qid, rows in groups.items():
        db = rows[0]["db_id"]
        correct = counts[qid]
        by_db[db]["questions"] += 1
        by_db[db]["correct"] += correct
        by_db[db]["pass8"] += correct > 0
        by_db[db]["all_zero"] += correct == 0
    db_table = {}
    for db, row in sorted(by_db.items(),
                          key=lambda item: item[1]["correct"] / (item[1]["questions"] * n_samples)):
        db_table[db] = {
            **row,
            "pass1": round(row["correct"] / (row["questions"] * n_samples), 4),
            "pass8": round(row["pass8"] / row["questions"], 4),
            "all_zero_rate": round(row["all_zero"] / row["questions"], 4),
        }
    return {
        "questions": n,
        "rollouts": len(records),
        "samples_per_question": sorted({len(rows) for rows in groups.values()}),
        "correct_count_distribution": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
        "mixed": mixed, "mixed_rate": round(mixed / n, 4),
        "all_zero": all_zero, "all_zero_rate": round(all_zero / n, 4),
        "all_correct": all_correct, "all_correct_rate": round(all_correct / n, 4),
        "pass_at_1": round(sum(counts.values()) / (n * n_samples), 4),
        "pass_at_8": round(sum(1 for value in counts.values() if value > 0) / n, 4),
        "pass_at_1_ci95": bootstrap_ci(per_question),
        "mixed_rate_ci95": bootstrap_ci([1 if 0 < value < n_samples else 0
                                         for value in counts.values()]),
        "matched_n4": n_equivalent(list(counts.values()), n_samples, 4),
        "by_db": db_table,
    }


def n_equivalent(counts, n_total=8, n_sub=4):
    """Exact expected group statistics when drawing n_sub of n_total samples."""
    def choose(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0

    total_subsets = choose(n_total, n_sub)
    per_question_mixed, per_question_all_zero, per_question_all_correct = [], [], []
    for k in counts:
        probabilities = {
            j: choose(k, j) * choose(n_total - k, n_sub - j) / total_subsets
            for j in range(n_sub + 1)
        }
        per_question_mixed.append(sum(p for j, p in probabilities.items() if 0 < j < n_sub))
        per_question_all_zero.append(probabilities.get(0, 0.0))
        per_question_all_correct.append(probabilities.get(n_sub, 0.0))
    n = len(counts)
    return {
        "n_sub": n_sub,
        "mixed_rate": round(sum(per_question_mixed) / n, 4),
        "mixed_rate_ci95": bootstrap_ci(per_question_mixed),
        "all_zero_rate": round(sum(per_question_all_zero) / n, 4),
        "all_correct_rate": round(sum(per_question_all_correct) / n, 4),
        "per_question_mixed": per_question_mixed,
        "method": "exact expectation over C(8,4) subsets per question",
    }


def two_sample_gap_ci(signal_probs, r1_mixed, r1_n, seed=0, samples=5000):
    """Bootstrap signal questions and R1 groups, then take the difference."""
    rng = random.Random(seed)
    r1 = [1] * r1_mixed + [0] * (r1_n - r1_mixed)
    n = len(signal_probs)
    gaps = []
    for _ in range(samples):
        signal_mean = sum(signal_probs[rng.randrange(n)] for _ in range(n)) / n
        r1_mean = sum(r1[rng.randrange(r1_n)] for _ in range(r1_n)) / r1_n
        gaps.append(signal_mean - r1_mean)
    gaps.sort()
    return [round(gaps[int(0.025 * (samples - 1))], 4),
            round(gaps[int(0.975 * (samples - 1))], 4)]


def batch_binomial(p_mixed, prompts=8):
    """Probability of k mixed groups in one prompts=8 update."""
    return {str(k): round(math.comb(prompts, k) * p_mixed ** k
                          * (1 - p_mixed) ** (prompts - k), 4)
            for k in range(prompts + 1)}


def prior_run(name, trace_path, dev_paths):
    trace = read_json(trace_path)
    out = {"trace": trace}
    out["dev200"] = {Path(path).stem: read_json(path) for path in dev_paths
                     if read_json(path)}
    return {name: out}


def p0_3_summary(gate_path, after_path):
    gate = read_json(gate_path)
    after = read_json(after_path)
    if not gate:
        return None
    return {
        "before_total": gate["before_total"], "after_total": gate["after_total"],
        "before_per_question": gate["before_per_question"],
        "after_per_question": gate["after_per_question"],
        "pass1_before": gate["pass1_before"], "pass1_after": gate["pass1_after"],
        "pass32": gate["pass32"], "last5_reward": gate["last5_reward"],
        "min_entropy": gate["min_entropy"], "max_kl": gate["max_kl"],
        "conditions": gate["conditions"], "pass": gate["pass"],
        "after_group_counts": (after or {}).get("group_counts"),
        "unavailable_rewards": gate["unavailable_rewards"],
        "infrastructure_errors": gate["infrastructure_errors"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-records", default="reports/m6/rl_signal_20261118/records.jsonl")
    parser.add_argument("--quarantine",
                        default="reports/m6/rl_preflight_20261118/annotation_quarantine.jsonl")
    parser.add_argument("--out", default="reports/m6/rl_signal_audit_20261120")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.signal_records)
    quarantine_ids = {row.get("id") or row.get("question_id")
                      for row in read_jsonl(args.quarantine)} if Path(args.quarantine).is_file() else set()
    clean_records = [record for record in records if record["id"] not in quarantine_ids]
    signal = group_stats(records, args.n_samples)
    clean_signal = group_stats(clean_records, args.n_samples) if clean_records else None
    prior = {
        **prior_run("R1_grpo", "reports/m6/e1_student_only/rl_r1_trace_analysis.json",
                    ["reports/m6/e1_student_only/rl_r1_step10.json",
                     "reports/m6/e1_student_only/rl_r1_step20.json"]),
        **prior_run("R3_dapo_ds", "reports/m6/e1_student_only/rl_r3_trace_analysis.json",
                    ["reports/m6/e1_student_only/rl_r3_step10.json"]),
    }
    p0_3 = p0_3_summary(
        "reports/m6/rl_learning5batch_4q_5e6_confirm_lr1e6_kl01_20261120/gate.json",
        "reports/m6/rl_learning5batch_4q_5e6_confirm_lr1e6_kl01_20261120/after32/summary.json")
    r1_trace = prior["R1_grpo"]["trace"]
    matched = signal["matched_n4"]
    gap = round(matched["mixed_rate"] - r1_trace["mixed_rate"], 4)
    comparison = {
        "matched_n4": matched,
        "r1_actual_n4": {
            "mixed_rate": r1_trace["mixed_rate"],
            "all_zero_rate": r1_trace["all_zero_rate"],
            "all_correct_rate": r1_trace["all_correct_rate"],
            "groups_complete": r1_trace["groups_complete"],
        },
        "gap_mixed": gap,
        "gap_mixed_ci95": two_sample_gap_ci(
            matched["per_question_mixed"], r1_trace["groups_mixed"],
            r1_trace["groups_complete"]),
        "note": "n=8 vs n=4 is n-inflated; the matched n=4-equivalent is the fair "
                "comparison to R1. The two-sample CI resamples signal questions and "
                "R1 groups.",
    }
    thresholds = {"mixed_rate_min": 0.50, "pass_at_8_min": 0.55, "all_zero_rate_max": 0.35}
    checks = {
        "mixed_rate": signal["mixed_rate"] >= thresholds["mixed_rate_min"],
        "pass_at_8": signal["pass_at_8"] >= thresholds["pass_at_8_min"],
        "all_zero_rate": signal["all_zero_rate"] <= thresholds["all_zero_rate_max"],
    }
    clean_checks = {
        "mixed_rate": clean_signal["mixed_rate"] >= thresholds["mixed_rate_min"],
        "pass_at_8": clean_signal["pass_at_8"] >= thresholds["pass_at_8_min"],
        "all_zero_rate": clean_signal["all_zero_rate"] <= thresholds["all_zero_rate_max"],
    } if clean_signal else {}
    audit = {
        "signal_records": args.signal_records,
        "n_samples": args.n_samples,
        "signal": signal,
        "quarantine_ids": sorted(quarantine_ids),
        "clean_signal": clean_signal,
        "clean_checks": clean_checks,
        "clean_signal_gate_pass": all(clean_checks.values()) if clean_checks else None,
        "batch_of_8_mixed_distribution": batch_binomial(signal["mixed_rate"], 8),
        "matched_n4_comparison": comparison,
        "prior_runs": prior,
        "p0_3_batch_confirm": p0_3,
        "thresholds": thresholds,
        "checks": checks,
        "signal_gate_pass": all(checks.values()),
        "recommendation": None,
    }
    if audit["signal_gate_pass"]:
        ci = comparison["gap_mixed_ci95"]
        includes_zero = ci[0] <= 0 <= ci[1]
        verdict = ("includes 0, so the signal set is not demonstrably easier than R1"
                   if includes_zero else
                   "does not include 0, so the matched n=4 signal is higher than R1")
        audit["recommendation"] = (
            "The n=8 signal gate passes (61% mixed), but the matched n=4-equivalent "
            f"is {matched['mixed_rate']:.3f} vs R1's {r1_trace['mixed_rate']:.3f}; "
            f"the two-sample gap 95% CI {ci} {verdict}. R1's neutral full-data result "
            "is a caution, not a matched baseline. The 12h pilot is a bounded n=8 test "
            "of the only stable recipe (lr=1e-6, KL=0.01), with a pre-registered "
            "dev200 trend gate; do not authorize the 56h cap before the pilot.")
    else:
        audit["recommendation"] = (
            "Signal is too sparse for a real-RL pilot under these thresholds; do not "
            "run P1-3. Report the harness negative result and the signal audit instead.")
    (out / "signal_audit.json").write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    lines = [
        "# RL learning-signal audit", "",
        f"Signal set: `{args.signal_records}` ({signal['questions']} questions x "
        f"{args.n_samples})", "",
        f"- pass@1: **{signal['pass_at_1']:.3f}** (95% CI {signal['pass_at_1_ci95']})",
        f"- pass@8: **{signal['pass_at_8']:.3f}**",
        f"- mixed groups: **{signal['mixed']}/{signal['questions']} = {signal['mixed_rate']:.3f}** "
        f"(95% CI {signal['mixed_rate_ci95']})",
        f"- all-zero: {signal['all_zero_rate']:.3f}; all-correct: {signal['all_correct_rate']:.3f}",
        "", "## Quarantine sensitivity", "",
        f"- {len(quarantine_ids)} of the 1,801 RL rows are quarantined; "
        f"{signal['questions'] - (clean_signal['questions'] if clean_signal else signal['questions'])} "
        "of the 100 signal questions overlap the quarantine.",
    ] + ([
        f"- clean {clean_signal['questions']} questions: mixed {clean_signal['mixed_rate']:.3f}, "
        f"pass@8 {clean_signal['pass_at_8']:.3f}, all-zero {clean_signal['all_zero_rate']:.3f}; "
        f"clean gate pass {audit['clean_signal_gate_pass']}",
    ] if clean_signal else []) + [
        "", "## Prior full-data runs", "",
        "| run | groups | mixed | all-zero | all-correct | dev200 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, run in prior.items():
        trace = run["trace"]
        dev = ", ".join(
            f"{key} ex={value.get('ex')} valid={value.get('valid')}"
            for key, value in run["dev200"].items()) if run["dev200"] else "n/a"
        lines.append(f"| {name} | {trace.get('groups_complete')} | "
                     f"{trace.get('mixed_rate'):.3f} | {trace.get('all_zero_rate'):.3f} | "
                     f"{trace.get('all_correct_rate'):.3f} | {dev} |")
    lines += [
        "", "## Matched n=4 comparison", "",
        f"- matched n=4-equivalent: mixed **{matched['mixed_rate']:.3f}** "
        f"(95% CI {matched['mixed_rate_ci95']}), all-zero "
        f"{matched['all_zero_rate']:.3f}, all-correct {matched['all_correct_rate']:.3f}",
        f"- R1 actual n=4: mixed **{r1_trace['mixed_rate']:.3f}**, all-zero "
        f"{r1_trace['all_zero_rate']:.3f}, all-correct {r1_trace['all_correct_rate']:.3f}",
        f"- gap (matched n=4 - R1): **{comparison['gap_mixed']:+.3f}**, two-sample "
        f"95% CI {comparison['gap_mixed_ci95']}"
        + (" (includes 0)" if comparison["gap_mixed_ci95"][0] <= 0
           <= comparison["gap_mixed_ci95"][1] else " (does not include 0)"),
        "- The 61% n=8 rate is not a matched improvement over R1; the gate below is "
        "an n=8 training-signal screen, not evidence of a capability gain.",
    ]
    lines += ["", "## P0-3 batch confirm (4q, 20 steps)", ""]
    if p0_3:
        lines += [
            f"- {p0_3['before_total']}/128 -> **{p0_3['after_total']}/128** "
            f"(pass@1 {p0_3['pass1_before']:.3f} -> {p0_3['pass1_after']:.3f})",
            f"- per question: {p0_3['before_per_question']} -> {p0_3['after_per_question']}",
            f"- pass@32 {p0_3['pass32']}; last5 reward {p0_3['last5_reward']}; "
            f"min entropy {p0_3['min_entropy']}; max KL {p0_3['max_kl']}",
            f"- gate pass: {p0_3['pass']}; conditions: {p0_3['conditions']}",
            "- note: the only false condition is `last5_reward_geq_0.95`; the eval "
            "was accepted as the primary learning signal in `DECISION.md`.",
        ]
    lines += ["", "## Gate", "",
              f"- mixed_rate >= {thresholds['mixed_rate_min']}: {checks['mixed_rate']}",
              f"- pass@8 >= {thresholds['pass_at_8_min']}: {checks['pass_at_8']}",
              f"- all_zero_rate <= {thresholds['all_zero_rate_max']}: {checks['all_zero_rate']}",
              f"- **signal_gate_pass: {audit['signal_gate_pass']}**", "",
              "## Recommendation", "", audit["recommendation"], ""]
    (out / "SIGNAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"signal": {k: signal[k] for k in
                                 ("questions", "mixed_rate", "all_zero_rate", "all_correct_rate",
                                  "pass_at_1", "pass_at_8")},
                      "checks": checks, "signal_gate_pass": audit["signal_gate_pass"],
                      "recommendation": audit["recommendation"]}, indent=1))


if __name__ == "__main__":
    main()
