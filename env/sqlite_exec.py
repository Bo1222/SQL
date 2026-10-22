"""Read-only SQLite execution with a query-level timeout."""
import os
import sqlite3
import time
from dataclasses import dataclass, field


@dataclass
class ExecResult:
    ok: bool
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    timed_out: bool = False
    truncated: bool = False


def _uri(path: str) -> str:
    return "file:" + os.path.abspath(path) + "?mode=ro&immutable=1"


def execute(db_path: str, sql: str, timeout_s: float = 5.0, max_rows: int = 1000) -> ExecResult:
    """Execute one statement against a read-only database snapshot."""
    start = time.time()
    con = None
    try:
        con = sqlite3.connect(_uri(db_path), uri=True, timeout=3)
        con.execute("PRAGMA query_only=ON")
        deadline = start + timeout_s
        con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return ExecResult(
            ok=True,
            rows=rows[:max_rows],
            columns=columns,
            elapsed_ms=(time.time() - start) * 1000,
            truncated=truncated,
        )
    except sqlite3.OperationalError as exc:
        message = str(exc)
        return ExecResult(
            ok=False,
            error=message,
            elapsed_ms=(time.time() - start) * 1000,
            timed_out="interrupt" in message.lower(),
        )
    except Exception as exc:  # noqa: BLE001
        return ExecResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.time() - start) * 1000,
        )
    finally:
        if con is not None:
            con.close()
