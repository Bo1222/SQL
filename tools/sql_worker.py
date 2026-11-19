"""Isolated stdlib-only SQLite worker. JSON on stdin, never a shell.

Read-only connection, authorizer-limited to SELECT/WITH, wall-clock progress
handler, historical project comparison (same length + same row set).
Official BIRD set-only comparison is available as a separate diagnostic;
the historical reward/evaluation contract is not silently changed.
"""
import json
import re
import resource
import sqlite3
import sys
import time
from pathlib import Path

SCORE_IMPLEMENTATION_VERSION = "bird-set-v3"
SCORE_MODES = {"project_strict": SCORE_IMPLEMENTATION_VERSION,
               "bird_set": "bird-official-set-v1"}


def infrastructure_exception(exc):
    text = str(exc).lower()
    return isinstance(exc, (MemoryError, OSError)) or any(
        w in text for w in ["malformed", "disk i/o", "unable to open", "not a database"])


def connection(path, timeout):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only=ON")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(operation, arg1, arg2, db, source):
        if operation not in allowed:
            return sqlite3.SQLITE_DENY
        if operation == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {
                "load_extension", "writefile", "readfile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    con.set_authorizer(authorize)
    return con


def query(con, sql):
    leading = re.sub(r"/\*.*?\*/|--[^\n]*", "", sql, flags=re.S).lstrip()
    if not re.match(r"(?:SELECT|WITH)\b", leading, re.I):
        raise ValueError("Only one SELECT/WITH query is permitted")
    return con.execute(sql)


def canonical(values):
    """Normalize a row so 1 == 1.0 and NULL is stable."""
    row = []
    for value in values:
        if value is None:
            row.append(("null",))
        elif isinstance(value, bool):
            row.append(("number", int(value), 1))
        elif isinstance(value, int):
            row.append(("number", value, 1))
        elif isinstance(value, float):
            import math
            row.append(("number", *value.as_integer_ratio()) if math.isfinite(value) else ("nonfinite", repr(value)))
        elif isinstance(value, bytes):
            row.append(("bytes", value.hex()))
        else:
            row.append(("text", str(value)))
    return tuple(row)


def compare_rows(gold_rows, pred_rows):
    """Historical project contract: equal row count + normalized row set."""
    gold = [canonical(r) for r in gold_rows]
    pred = [canonical(r) for r in pred_rows]
    return len(gold) == len(pred) and set(gold) == set(pred)


def run(request):
    started = time.monotonic()
    def stage(name):
        if request.get("report_stage"):
            print("SQLRL_STAGE=" + name, file=sys.stderr, flush=True)

    def failure(exc, origin, **fields):
        database_error = not isinstance(exc, MemoryError) and infrastructure_exception(exc)
        if database_error and origin == "candidate":
            origin = "database"
        return {"ok": False, "correct": False, "error": str(exc) or type(exc).__name__,
                "exception_type": type(exc).__name__, "failure_origin": origin,
                "reward_available": origin == "candidate",
                **({"infrastructure_error": True} if origin != "candidate" or infrastructure_exception(exc) else {}),
                **fields}

    stage("database")
    try:
        con = connection(request["db_path"], request.get("timeout", 30))
    except (sqlite3.Error, OSError) as exc:
        return failure(exc, "database")
    try:
        operation = request["operation"]
        if operation in {"score", "score_diagnostic"}:
            score_mode = request.get("score_mode", "project_strict")
            if score_mode not in SCORE_MODES:
                return {"ok": False, "correct": False, "infrastructure_error": True,
                        "error": f"unknown score mode: {score_mode}"}
            try:
                stage("gold")
                gold = list(query(con, request["gold_sql"]))
            except (sqlite3.Error, ValueError, MemoryError, OSError) as exc:
                return failure(exc, "gold", gold_ok=False, gold_error=str(exc))
            try:
                stage("candidate")
                pred = list(query(con, request["sql"]))
            except (sqlite3.Error, ValueError, MemoryError, OSError) as exc:
                return failure(exc, "candidate", gold_ok=True)
            project_correct = compare_rows(gold, pred)
            bird_correct = set(gold) == set(pred)
            correct = project_correct if score_mode == "project_strict" else bird_correct
            result = {"ok": True, "correct": correct, "gold_ok": True,
                      "reward_available": True, "failure_origin": None,
                      "gold_rows": len(gold), "pred_rows": len(pred),
                      "score_mode": score_mode, "score_implementation_version": SCORE_MODES[score_mode],
                      "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
            if operation == "score_diagnostic":
                result.update({"project_strict_correct": project_correct,
                               "bird_set_only_correct": bird_correct,
                               "canonical_set_only_correct": set(map(canonical, gold)) == set(map(canonical, pred))})
            return result
        stage("candidate")
        cursor = query(con, request["sql"])
        limit = min(int(request.get("limit", 3)), 20)
        rows = cursor.fetchmany(limit + 1)
        return {"ok": True, "columns": [c[0] for c in cursor.description],
                "preview": rows[:limit], "has_more": len(rows) > limit,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
    except (sqlite3.Error, ValueError, MemoryError, OSError) as exc:
        return failure(exc, "candidate")
    finally:
        con.close()


if __name__ == "__main__":
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
        result = run(json.load(sys.stdin))
    except (sqlite3.Error, ValueError, MemoryError, OSError) as exc:
        result = {"ok": False, "infrastructure_error": isinstance(exc, (MemoryError, OSError)), "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False))
