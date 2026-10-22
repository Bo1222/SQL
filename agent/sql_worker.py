"""Isolated stdlib-only SQLite worker. Called with JSON on stdin, never a shell."""
import json
import os
import re
import resource
import sqlite3
import sys
import time


def connection(path, timeout):
    from pathlib import Path
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only=ON")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               sqlite3.SQLITE_RECURSIVE}
    def authorize(operation, arg1, arg2, db, source):
        if operation not in allowed or (operation == sqlite3.SQLITE_FUNCTION and
                                       (arg2 or "").lower() in {"load_extension", "writefile", "readfile"}):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    con.set_authorizer(authorize)
    return con


def query(con, sql):
    # SQLite itself validates single statements and the authorizer rejects writable CTEs.
    leading = re.sub(r"/\*.*?\*/|--[^\n]*", "", sql, flags=re.S).lstrip()
    if not re.match(r"(?:SELECT|WITH)\b", leading, re.I):
        raise ValueError("Only one SELECT/WITH query is permitted")
    return con.execute(sql)


def run(request):
    started = time.monotonic()
    con = connection(request["db_path"], request.get("timeout", 30))
    try:
        operation = request["operation"]
        if operation == "signature":
            import hashlib
            rows = set(query(con, request["sql"]).fetchall())
            def canonical(value):
                # Python/BIRD considers integer 1 and float 1.0 equal across queries.
                if value is None: return ("null",)
                if isinstance(value, int): return ("number", value, 1)
                if isinstance(value, float):
                    import math
                    return ("number", *value.as_integer_ratio()) if math.isfinite(value) else ("nonfinite", repr(value))
                if isinstance(value, bytes): return ("bytes", value.hex())
                return ("text", value)
            normalized = sorted(tuple(canonical(v) for v in row) for row in rows)
            signature = hashlib.sha256(repr(normalized).encode()).hexdigest()
            return {"ok": True, "signature": signature}
        if operation == "score":
            # Same set-of-row-tuples semantics as BIRD's public evaluator. No row cap.
            gold = set(query(con, request["gold_sql"]).fetchall())
            try:
                pred = set(query(con, request["sql"]).fetchall())
                return {"ok": True, "correct": pred == gold, "gold_ok": True,
                        "gold_rows": len(gold), "pred_rows": len(pred)}
            except (sqlite3.Error, ValueError) as exc:
                return {"ok": False, "correct": False, "gold_ok": True, "error": str(exc)}
        cursor = query(con, request["sql"])
        if operation == "validate":
            while cursor.fetchmany(4096):
                pass
            return {"ok": True}
        limit = min(int(request.get("limit", 5)), 20)
        rows = cursor.fetchmany(limit + 1)
        return {"ok": True, "columns": [c[0] for c in cursor.description],
                "preview": rows[:limit], "preview_rows": min(len(rows), limit),
                "has_more": len(rows) > limit, "row_count_is_exact": len(rows) <= limit}
    finally:
        con.close()


if __name__ == "__main__":
    try:
        # Bound a runaway result materialization as well as SQL execution time.
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
        request = json.load(sys.stdin)
        result = run(request)
    except (sqlite3.Error, ValueError, MemoryError, OSError) as exc:
        result = {"ok": False, "correct": False, "gold_ok": False,
                  "infrastructure_error": isinstance(exc, (MemoryError, OSError)), "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False))
