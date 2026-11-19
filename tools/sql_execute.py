"""Read-only SQL execution tool with structured error classification."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERROR_TYPES = [
    ("interrupt", "timeout"), ("one statement at a time", "multi_statement"),
    ("no such table", "no_such_table"), ("no such column", "no_such_column"),
    ("no such function", "no_such_function"), ("ambiguous column", "ambiguous_column"),
    ("syntax error", "syntax_error"), ("misuse", "misuse"),
    ("datatype mismatch", "datatype_mismatch"), ("unable to open", "database_error"),
]


def error_type(message):
    if not message:
        return "none"
    text = str(message).lower()
    for needle, name in ERROR_TYPES:
        if needle in text:
            return name
    return "other"


def run_sql(db_path, sql, operation="preview", timeout=10, gold_sql=None, limit=3,
            score_mode="project_strict", report_stage=False):
    """Execute SQL in an isolated subprocess; never raises for SQL-level errors."""
    started = time.monotonic()
    if not Path(db_path).is_file():
        return {"ok": False, "infrastructure_error": True, "error": "database missing",
                "failure_origin": "database", "reward_available": False}
    request = dict(db_path=str(Path(db_path).resolve()), sql=sql, operation=operation,
                   timeout=timeout, limit=limit, gold_sql=gold_sql, score_mode=score_mode,
                   report_stage=report_stage)
    try:
        process = subprocess.run([sys.executable, "-m", "tools.sql_worker"], input=json.dumps(request),
                                 text=True, capture_output=True, cwd=str(ROOT), timeout=timeout + 5)
        if process.returncode:
            result = {"ok": False, "infrastructure_error": True, "raw_error": process.stderr[-500:],
                      "error": f"SQL worker exited {process.returncode}",
                      "failure_origin": "worker", "reward_available": False}
        else:
            result = json.loads(process.stdout)
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        phases = [line.split("=", 1)[1] for line in stderr.splitlines() if line.startswith("SQLRL_STAGE=")]
        origin = phases[-1] if report_stage and phases else "worker"
        result = {"ok": False, "error": "worker watchdog timeout", "timed_out": True,
                  "infrastructure_error": origin != "candidate", "failure_origin": origin,
                  "reward_available": origin == "candidate"}
    except (OSError, ValueError, TypeError) as exc:
        result = {"ok": False, "infrastructure_error": True, "error": f"worker transport failure: {exc}",
                  "failure_origin": "worker", "reward_available": False}
    result["wall_ms"] = round((time.monotonic() - started) * 1000, 2)
    if not result.get("ok") and "error" in result and not result.get("infrastructure_error"):
        result["error_type"] = "timeout" if result.get("timed_out") else error_type(result.get("error"))
    return result


class SQLExecuteTool:
    def __init__(self, db_path, timeout=10, preview_rows=3, failure_policy="legacy"):
        self.db_path, self.timeout, self.preview_rows = db_path, timeout, preview_rows
        self.failure_policy = failure_policy

    def __call__(self, sql):
        typed = self.failure_policy == "typed-v1"
        result = run_sql(self.db_path, sql, timeout=self.timeout, limit=self.preview_rows,
                         report_stage=typed)
        if result.get("ok"):
            return {"status": "ok", "columns": result.get("columns", []),
                    "rows_preview": result.get("preview", []),
                    "truncated": bool(result.get("has_more")),
                    "elapsed_ms": result.get("elapsed_ms")}
        kind = result.get("error_type") or "database_error"
        if kind in {"no_such_column", "no_such_table", "ambiguous_column"}:
            kind = "schema_error"
        elif kind not in {"syntax_error", "timeout"}:
            kind = "database_error"
        return {"status": "error", "error_type": kind, "message": str(result.get("error"))[:300],
                **({"failure_origin": "infrastructure"} if (
                    not result.get("reward_available", False) if typed else result.get("infrastructure_error")) else {})}
