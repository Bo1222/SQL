"""Terminal BIRD EX, completely separated from model-visible tools."""
from agent.tools import run_sql
from reward.exec_reward import FROM_RE

SCORER_VERSION = "bird-public-set-v1-full-results"


def score_trajectory(row, trajectory, timeout=30):
    if trajectory.stop_reason == "infrastructure_error":
        return {"reward": 0.0, "correct": False, "ok": False, "infrastructure_error": True}
    if trajectory.final_sql is None:
        return {"reward": 0.0, "correct": False, "ok": False, "error": trajectory.stop_reason}
    result = run_sql(row["db_path"], trajectory.final_sql, "score", timeout, row["gold_sql"])
    if result.get("gold_ok") is False:
        result["infrastructure_error"] = True
    correct = bool(result.get("correct"))
    hard = ["constant_no_from"] if correct and FROM_RE.search(row["gold_sql"]) and not FROM_RE.search(trajectory.final_sql) else []
    return {**result, "reward": float(correct), "hard_flags": hard,
            "ex_clean": bool(correct and not hard), "scorer_version": SCORER_VERSION}
