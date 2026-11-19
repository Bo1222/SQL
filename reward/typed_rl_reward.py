"""Fixed-schema adapter for veRL's batched structured-reward collector.

Core grading stays in bird_ex_reward. veRL indexes every sample with the first
sample's reward keys, so missing-answer and grader-failure results must expose
the same fields as normally executed SQL.
"""
from reward.bird_ex_reward import compute_score_with_details as _grade

DEFAULTS = {
    "score": 0.0, "final_sql": None, "correct": False, "ok": False,
    "error": None, "error_type": None, "infrastructure_error": False,
    "failure_origin": None, "reward_available": False, "gold_error": None,
    "score_mode": "bird_set", "score_implementation_version": None,
    "gold_rows": None, "pred_rows": None, "reward_contract": "typed-v1",
    "score_attempts": 1, "first_grader_failure": "",
}


def compute_score_with_details(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not extra_info or extra_info.get("score_mode") != "bird_set":
        raise ValueError("typed RL requires explicit bird_set scoring")
    result = _grade(data_source, solution_str, ground_truth, extra_info, **kwargs)
    unknown = result.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"new reward fields require a schema update: {sorted(unknown)}")
    return {**DEFAULTS, **result}
