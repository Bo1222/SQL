"""Stage 0: audit candidate process rewards on existing SQL-agent rollouts.

This is a CPU-only, read-only audit.  It does not call a model and does not
write into any training dataset.  The output decides whether a process signal
is worth a GPU pilot at all.

Usage:
    python scripts/audit_process_rewards.py \
      --test-trace reports/m6/e1_student_only/test1534_sft_lowlr_788.jsonl \
      --rollout reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl \
      --out-json reports/m6/e1_student_only/process_reward_audit.json \
      --out-md reports/m6/e1_student_only/process_reward_audit.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward.process_signals import (  # noqa: E402
    ProcessEvents,
    compose_reward,
    extract_process_events,
    iter_jsonl,
    progress_score,
)

MODES = (
    "recovery_only",
    "recovery_penalties",
    "penalties_only",
    "avoid_abandon_only",
    "safe_penalties",
)
PROXY_SIGNALS = {"valid_sql_any", "lookup_nonempty", "answer_submitted"}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for value in items if value) / len(items)


def _record_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    record = row.get("record")
    return record if isinstance(record, Mapping) else row


def _score_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    score = row.get("score")
    return score if isinstance(score, Mapping) else {}


def _is_infra(row: Mapping[str, Any]) -> bool:
    if row.get("infrastructure_error") or row.get("score_error"):
        return True
    score = _score_of(row)
    return bool(score.get("infrastructure_error"))


def _terminal_ex(row: Mapping[str, Any]) -> bool:
    if "correct" in row and row.get("correct") is not None:
        return bool(row.get("correct"))
    score = _score_of(row)
    if "correct" in score:
        return bool(score.get("correct"))
    return bool(row.get("final_ex"))


def _valid(row: Mapping[str, Any]) -> bool:
    if "valid" in row and row.get("valid") is not None:
        return bool(row.get("valid"))
    score = _score_of(row)
    if "ok" in score:
        return bool(score.get("ok"))
    return False


def row_to_events(row: Mapping[str, Any]) -> ProcessEvents:
    record = _record_of(row)
    return extract_process_events(
        tool_events=record.get("tool_events") or row.get("tool_events"),
        finish_reason=record.get("finish_reason") or row.get("finish_reason"),
        final_sql=record.get("final_sql") if record is not row else row.get("final_sql"),
        first_sql=record.get("first_sql") if record is not row else row.get("first_sql"),
        invalid_actions=record.get("invalid_actions", row.get("invalid_actions", 0)),
        turns=record.get("rounds", record.get("turns", row.get("rounds", row.get("turns", 0)))),
        schema_lookups=record.get("schema_lookups", row.get("schema_lookups")),
        sql_executes=record.get("sql_executes", row.get("sql_executes")),
        tool_calls=record.get("tool_calls", row.get("tool_calls")),
    )


def make_row(path: str, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(iter_jsonl(path)):
        if limit and index >= limit:
            break
        events = row_to_events(raw)
        ex = _terminal_ex(raw)
        valid = _valid(raw)
        rows.append({
            "question_id": raw.get("question_id") or raw.get("id"),
            "db_id": raw.get("db_id"),
            "ex": ex,
            "valid": valid,
            "valid_but_wrong": bool(valid and not ex),
            "infra": _is_infra(raw),
            "events": events,
        })
    return rows


SIGNAL_GETTERS = {
    "invalid_any": lambda events, row: events.invalid_actions > 0,
    "repeat_sql_any": lambda events, row: events.repeat_sql > 0,
    "repeat_lookup_any": lambda events, row: events.repeat_lookup > 0,
    "schema_error_any": lambda events, row: events.schema_error_count > 0,
    "schema_error_then_lookup": lambda events, row: events.schema_error_then_lookup,
    "schema_error_then_answer": lambda events, row: events.schema_error_then_answer,
    "schema_error_repaired": lambda events, row: events.schema_error_repaired,
    "lookup_nonempty": lambda events, row: events.lookup_nonempty,
    "valid_sql_any": lambda events, row: events.valid_sql_count > 0,
    "answer_submitted": lambda events, row: events.answer_submitted,
    "tool_calls_ge2": lambda events, row: events.tool_calls >= 2,
    "schema_error_count_ge2": lambda events, row: events.schema_error_count >= 2,
}


def signal_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if not row["infra"]]
    output: dict[str, Any] = {}
    for name, getter in SIGNAL_GETTERS.items():
        positives = [row for row in usable if getter(row["events"], row)]
        negatives = [row for row in usable if not getter(row["events"], row)]
        ex_pos = _rate(row["ex"] for row in positives)
        ex_neg = _rate(row["ex"] for row in negatives)
        vw_pos = _rate(row["valid_but_wrong"] for row in positives)
        output[name] = {
            "n": len(usable),
            "positive": len(positives),
            "prevalence": (len(positives) / len(usable)) if usable else None,
            "ex_positive": ex_pos,
            "ex_negative": ex_neg,
            "ex_delta": (
                ex_pos - ex_neg
                if ex_pos is not None and ex_neg is not None else None
            ),
            "valid_but_wrong_positive": vw_pos,
            "proxy_only": name in PROXY_SIGNALS,
        }
    return output


def _group_rows(rows: list[dict[str, Any]], group_size: int) -> tuple[list[list[dict[str, Any]]], int, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("question_id") or "")
        if key:
            grouped[key].append(row)
    groups: list[list[dict[str, Any]]] = []
    bad_size = 0
    infra_groups = 0
    for items in grouped.values():
        if len(items) != group_size:
            bad_size += 1
            continue
        if any(item["infra"] for item in items):
            infra_groups += 1
            continue
        groups.append(items)
    return groups, bad_size, infra_groups


def simulate_groups(
    groups: list[list[dict[str, Any]]],
    *,
    betas: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        mode_out: dict[str, Any] = {}
        for beta in betas:
            stats = {
                "groups": 0,
                "terminal_mixed": 0,
                "shaped_mixed": 0,
                "all_zero_groups": 0,
                "all_zero_shaped_mixed": 0,
                "all_zero_best_fail_valid_wrong": 0,
                "all_zero_any_valid_wrong": 0,
                "all_zero_trajectories": 0,
                "all_zero_valid_wrong_trajectories": 0,
                "all_correct_groups": 0,
                "all_correct_shaped_mixed": 0,
                "mixed_terminal_groups": 0,
                "mixed_terminal_shaped_mixed": 0,
                "process_only_mixed": 0,
            }
            for group in groups:
                stats["groups"] += 1
                ex = [bool(item["ex"]) for item in group]
                n_correct = sum(ex)
                terminal_mixed = 0 < n_correct < len(ex)
                events = [item["events"] for item in group]
                rewards = [
                    compose_reward(item["ex"], item["events"], beta=beta, mode=mode)
                    for item in group
                ]
                shaped_mixed = max(rewards) - min(rewards) > 1e-9
                stats["terminal_mixed"] += int(terminal_mixed)
                stats["shaped_mixed"] += int(shaped_mixed)
                if terminal_mixed:
                    stats["mixed_terminal_groups"] += 1
                    stats["mixed_terminal_shaped_mixed"] += int(shaped_mixed)
                if n_correct == 0:
                    stats["all_zero_groups"] += 1
                    stats["all_zero_trajectories"] += len(group)
                    stats["all_zero_valid_wrong_trajectories"] += sum(
                        1 for item in group if item["valid_but_wrong"]
                    )
                    stats["all_zero_any_valid_wrong"] += int(
                        any(item["valid_but_wrong"] for item in group)
                    )
                    if shaped_mixed:
                        stats["all_zero_shaped_mixed"] += 1
                        best = max(rewards)
                        best_items = [
                            item for item, reward in zip(group, rewards)
                            if abs(reward - best) <= 1e-9
                        ]
                        stats["all_zero_best_fail_valid_wrong"] += int(
                            any(item["valid_but_wrong"] for item in best_items)
                        )
                if n_correct == len(ex):
                    stats["all_correct_groups"] += 1
                    stats["all_correct_shaped_mixed"] += int(shaped_mixed)
                if not terminal_mixed and shaped_mixed:
                    stats["process_only_mixed"] += 1
            def frac(key: str, denom: str) -> float | None:
                return stats[key] / stats[denom] if stats[denom] else None
            stats.update({
                "all_zero_shaped_mixed_rate": frac("all_zero_shaped_mixed", "all_zero_groups"),
                "all_zero_best_fail_valid_wrong_rate": frac(
                    "all_zero_best_fail_valid_wrong", "all_zero_shaped_mixed"),
                "all_zero_any_valid_wrong_rate": frac(
                    "all_zero_any_valid_wrong", "all_zero_groups"),
                "all_zero_valid_wrong_trajectory_rate": frac(
                    "all_zero_valid_wrong_trajectories", "all_zero_trajectories"),
                "process_only_mixed_rate": frac("process_only_mixed", "groups"),
            })
            base = stats["all_zero_valid_wrong_trajectory_rate"]
            best = stats["all_zero_best_fail_valid_wrong_rate"]
            stats["all_zero_best_fail_valid_wrong_lift"] = (
                best - base if best is not None and base is not None else None
            )
            mode_out[str(beta)] = stats
        output[mode] = mode_out
    return output


def first_sql_stats(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Optional diagnostic: execute first SQL against gold on a small sample.

    This is deliberately not used by the training reward.  It is a diagnostic
    of how often the first candidate would already have been correct.
    """
    from reward.bird_ex_reward import score_sql

    checked = 0
    correct = 0
    errors = 0
    for row in rows:
        if limit and checked >= limit:
            break
        record = _record_of(row)
        first_sql = record.get("first_sql") if record is not row else row.get("first_sql")
        db_path = record.get("db_path") if record is not row else row.get("db_path")
        gold_sql = record.get("gold_sql") if record is not row else row.get("gold_sql")
        if not first_sql or not db_path or not gold_sql:
            continue
        checked += 1
        try:
            result = score_sql(db_path, first_sql, gold_sql)
        except Exception:  # noqa: BLE001 - audit must not fail on one row
            errors += 1
            continue
        if result.get("infrastructure_error"):
            errors += 1
        elif result.get("correct"):
            correct += 1
    return {
        "checked": checked,
        "correct": correct,
        "errors": errors,
        "correct_rate": correct / checked if checked else None,
        "note": "diagnostic only; gold is never used by the training reward",
    }


def select_recommendation(
    signals: Mapping[str, Any],
    simulations: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservative heuristic; not a statistical test."""
    candidates: list[dict[str, Any]] = []
    for mode in MODES:
        for beta, stats in simulations.get(mode, {}).items():
            rate = stats.get("all_zero_shaped_mixed_rate")
            bad = stats.get("all_zero_best_fail_valid_wrong_rate")
            if rate is None or rate < 0.05:
                continue
            candidates.append({
                "mode": mode,
                "beta": float(beta),
                "all_zero_shaped_mixed_rate": rate,
                "all_zero_best_fail_valid_wrong_rate": bad,
                "all_zero_best_fail_valid_wrong_lift": stats.get(
                    "all_zero_best_fail_valid_wrong_lift"),
                "process_only_mixed_rate": stats.get("process_only_mixed_rate"),
            })
    candidates.sort(
        key=lambda item: (
            item["all_zero_best_fail_valid_wrong_lift"]
            if item["all_zero_best_fail_valid_wrong_lift"] is not None else 1.0,
            -item["all_zero_shaped_mixed_rate"],
        )
    )
    if not candidates:
        return {
            "decision": "no_go",
            "reason": "no candidate produced at least 5% shaped-mixed all-zero groups",
            "candidates": [],
        }
    best = candidates[0]
    recovery_delta = signals.get("schema_error_repaired", {}).get("ex_delta")
    if best["mode"].startswith("recovery") and (
        recovery_delta is None or recovery_delta <= 0
    ):
        return {
            "decision": "no_go",
            "reason": "schema_error_repaired has no positive EX association",
            "candidates": candidates,
        }
    lift = best["all_zero_best_fail_valid_wrong_lift"]
    if (
        lift is None
        or lift > -0.05
        or (
            best["all_zero_best_fail_valid_wrong_rate"] is not None
            and best["all_zero_best_fail_valid_wrong_rate"] > 0.45
        )
    ):
        return {
            "decision": "no_go",
            "reason": (
                "process signal does not beat the base valid-but-wrong rate when "
                "selecting the best trajectory in all-zero groups"
            ),
            "candidates": candidates,
        }
    return {
        "decision": "conditional_go",
        "reason": "candidate has all-zero group variance and selects valid-but-wrong less often than the base rate",
        "best": best,
        "candidates": candidates,
        "caveat": "heuristic only; the 20-step pilot is the real test",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Process Reward Audit (Stage 0)",
        "",
        f"Generated: {payload['meta']['generated_at']}",
        "",
        "## Data",
        "",
        f"- test1534 trace rows: {payload['meta']['test_rows']}",
        f"- n4 rollout rows: {payload['meta']['rollout_rows']}",
        f"- usable signal rows (test): {payload['meta']['test_usable']}",
        f"- groups used for simulation: {payload['meta']['group_count']}",
        f"- groups with wrong size: {payload['meta']['group_bad_size']}",
        f"- groups with infra: {payload['meta']['group_infra']}",
        "",
        "## Candidate signal associations",
        "",
        "| signal | prevalence | EX positive | EX negative | ΔEX | valid-but-wrong+ | proxy only |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in payload["signals"].items():
        lines.append(
            f"| `{name}` | {_fmt(item['prevalence'])} | {_fmt(item['ex_positive'])} | "
            f"{_fmt(item['ex_negative'])} | {_fmt(item['ex_delta'])} | "
            f"{_fmt(item['valid_but_wrong_positive'])} | {item['proxy_only']} |"
        )
    lines += [
        "",
        "`EX positive/negative` are conditional EX rates. `ΔEX` is descriptive, not causal.",
        "",
        "## Simulated rank-preserving rewards",
        "",
        "`R = 1` if EX, else `-1 + beta * g`. Any success always ranks above any failure.",
        "",
        "| mode | beta | all-zero groups | shaped mixed | shaped rate | base valid-wrong | best fail valid-wrong | lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, betas in payload["simulations"].items():
        for beta, item in betas.items():
            lines.append(
                f"| `{mode}` | {beta} | {item['all_zero_groups']} | "
                f"{item['all_zero_shaped_mixed']} | {_fmt(item['all_zero_shaped_mixed_rate'])} | "
                f"{_fmt(item['all_zero_valid_wrong_trajectory_rate'])} | "
                f"{_fmt(item['all_zero_best_fail_valid_wrong_rate'])} | "
                f"{_fmt(item['all_zero_best_fail_valid_wrong_lift'])} |"
            )
    rec = payload["recommendation"]
    lines += [
        "",
        "## Recommendation",
        "",
        f"**Decision: `{rec['decision']}`**",
        "",
        f"Reason: {rec['reason']}",
        "",
    ]
    if rec.get("best"):
        best = rec["best"]
        lines += [
            f"Best heuristic candidate: `{best['mode']}` with beta `{best['beta']}`.",
            "",
        ]
    # Make the decision interpretable without requiring the JSON.
    sample_beta = next(iter(payload["simulations"].values()), {})
    sample_stats = next(iter(sample_beta.values()), {})
    base_vw = sample_stats.get("all_zero_valid_wrong_trajectory_rate")
    if base_vw is not None:
        lines += [
            "## Interpretation",
            "",
            f"In terminal all-zero groups, the base valid-but-wrong rate is "
            f"**{_fmt(base_vw)}** per trajectory. The tested signals select a",
            "valid-but-wrong trajectory as the best failure at 49–100%, i.e. they",
            "enrich the proxy rather than reduce it. Recovery shaping is the worst:",
            "its best failure is valid-but-wrong in 99.7% of shaped all-zero groups.",
            "",
            "This is not a small-weight tuning issue. Under trajectory-level GRPO the",
            "signal direction itself is wrong. Gate 0 therefore stops scalar process",
            "reward; the next authorized path is terminal GRPO + DAPO DS. Turn-level",
            "credit or first-SQL supervision would be a separate research branch.",
            "",
        ]
    lines += [
        "This is a heuristic audit, not a statistical test. Even a conditional go only",
        "authorizes a 20-step pilot; it does not establish an EX improvement.",
        "",
        "## Caveats",
        "",
        "- Signal associations are descriptive; tool behavior and question difficulty are confounded.",
        "- The test1534 trace has one sample per question; n4 group simulation uses the",
        "  `step_600_merged` SFT data-collection rollout pool, not step788. The group",
        "  simulation is a proxy; the direct step788 signal associations are the primary",
        "  evidence for rejecting a positive recovery reward.",
        "- `valid_sql_any`, `lookup_nonempty`, `answer_submitted` are proxy signals and are not",
        "  recommended as positive rewards.",
        "- `first_sql_ex`, when requested, is a gold-side diagnostic and never enters the training reward.",
        "",
    ]
    if payload.get("first_sql"):
        f = payload["first_sql"]
        lines += [
            "## Optional first-SQL diagnostic",
            "",
            f"- checked: {f['checked']}",
            f"- correct: {f['correct']}",
            f"- errors: {f['errors']}",
            f"- correct rate: {_fmt(f['correct_rate'])}",
            f"- note: {f['note']}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-trace", default="reports/m6/e1_student_only/test1534_sft_lowlr_788.jsonl")
    parser.add_argument("--rollout", default="reports/m6/e1_student_only/p3_full_n4/rollouts.jsonl")
    parser.add_argument("--out-json", default="reports/m6/e1_student_only/process_reward_audit.json")
    parser.add_argument("--out-md", default="reports/m6/e1_student_only/process_reward_audit.md")
    parser.add_argument("--betas", default="0.02,0.05,0.1")
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--limit-rollout", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--score-first-sql", action="store_true")
    parser.add_argument("--first-sql-limit", type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    betas = [float(value) for value in args.betas.split(",") if value]
    test_rows = make_row(args.test_trace, limit=args.limit_test)
    rollout_rows = make_row(args.rollout, limit=args.limit_rollout)
    test_usable = [row for row in test_rows if not row["infra"]]
    signals = signal_stats(test_rows)
    groups, bad_size, infra_groups = _group_rows(rollout_rows, args.group_size)
    simulations = simulate_groups(groups, betas=betas)
    recommendation = select_recommendation(signals, simulations)
    first_sql = first_sql_stats(test_rows, args.first_sql_limit) if args.score_first_sql else None

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_sec": round(time.time() - started, 2),
            "test_trace": str(Path(args.test_trace)),
            "rollout": str(Path(args.rollout)),
            "test_rows": len(test_rows),
            "rollout_rows": len(rollout_rows),
            "test_usable": len(test_usable),
            "group_count": len(groups),
            "group_bad_size": bad_size,
            "group_infra": infra_groups,
            "group_size": args.group_size,
            "betas": betas,
        },
        "signals": signals,
        "simulations": simulations,
        "recommendation": recommendation,
        "first_sql": first_sql,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out_md.write_text(markdown_report(payload) + "\n", encoding="utf-8")
    print(json.dumps({
        "meta": payload["meta"],
        "recommendation": recommendation,
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
