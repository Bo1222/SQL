"""BIRD execution-accuracy reward for veRL's custom_reward_function.

Signature follows veRL: compute_score(data_source, solution_str, ground_truth, extra_info).
Gold SQL is only ever used inside the isolated SQL worker; it never enters the
prompt or an observation.
"""
import json
import re

from tools.sql_execute import run_sql

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_final_sql(text):
    """Last complete <answer> tag; training rewards never guess a missing answer."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    sql = matches[-1].strip()
    return sql or None


def score_sql(db_path, candidate_sql, gold_sql, timeout=30, score_mode="project_strict", typed=False):
    if not candidate_sql:
        return {"correct": False, "ok": False, "error": "missing_answer", "infrastructure_error": False,
                "failure_origin": "candidate", "reward_available": True}
    result = run_sql(db_path, candidate_sql, operation="score", timeout=timeout,
                     gold_sql=gold_sql, score_mode=score_mode, report_stage=typed)
    correct = bool(result.get("correct"))
    return {"correct": correct, "ok": bool(result.get("ok")),
            "error": result.get("error"), "error_type": result.get("error_type"),
            "infrastructure_error": bool(result.get("infrastructure_error")),
            "failure_origin": result.get("failure_origin"),
            "reward_available": bool(result.get("reward_available", not result.get("infrastructure_error"))),
            "gold_error": result.get("gold_error"), "score_mode": score_mode,
            "score_implementation_version": result.get("score_implementation_version"),
            "gold_rows": result.get("gold_rows"), "pred_rows": result.get("pred_rows")}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra = extra_info or {}
    db_path = extra.get("db_path")
    sql = extract_final_sql(solution_str)
    scored = score_sql(db_path, sql, ground_truth, score_mode=extra.get("score_mode", "project_strict"))
    if scored["infrastructure_error"]:
        # Historical scalar API: cannot express an unavailable reward. New RL
        # must use compute_score_with_details plus whole-group filtering.
        return 0.0
    return 1.0 if scored["correct"] else 0.0


def compute_score_with_details(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra = extra_info or {}
    sql = extract_final_sql(solution_str)
    scored = score_sql(extra.get("db_path"), sql, ground_truth,
                       score_mode=extra.get("score_mode", "project_strict"), typed=True)
    attempts = 1
    first_failure = None
    if not scored["reward_available"]:
        first_failure = {k: scored.get(k) for k in ("failure_origin", "error", "gold_error")}
        scored = score_sql(extra.get("db_path"), sql, ground_truth,
                           score_mode=extra.get("score_mode", "project_strict"), typed=True)
        attempts += 1
    available = bool(scored["reward_available"])
    return {"score": float(scored["correct"]) if available else 0.0,
            "final_sql": sql, **scored, "reward_available": available,
            "reward_contract": "typed-v1", "score_attempts": attempts,
            "first_grader_failure": json.dumps(first_failure, ensure_ascii=False) if first_failure else ""}

