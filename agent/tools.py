"""Metadata retrieval, bounded value inspection, and isolated read-only SQL."""
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from reward.exec_reward import error_type

ROOT = Path(__file__).resolve().parents[1]


def run_sql(db_path, sql, operation="preview", timeout=10, gold_sql=None, limit=5):
    started = time.monotonic()
    if not Path(db_path).is_file():
        return {"ok": False, "gold_ok": False, "infrastructure_error": True, "error": "database missing"}
    request = dict(db_path=str(Path(db_path).resolve()), sql=sql, operation=operation,
                   timeout=timeout, limit=limit, gold_sql=gold_sql)
    try:
        process = subprocess.run([sys.executable, "-m", "agent.sql_worker"],
                                 input=json.dumps(request), text=True, capture_output=True,
                                 cwd=ROOT, timeout=timeout + 2)
        if process.returncode:
            result = {"ok": False, "gold_ok": False, "infrastructure_error": True,
                      "error": f"SQL worker exited {process.returncode}"}
        else:
            result = json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        result = {"ok": False, "correct": False, "error": "query timeout", "timed_out": True}
        if operation == "score":
            result.update(gold_ok=False, infrastructure_error=True)
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
    if not result["ok"]:
        result["error_type"] = "timeout" if result.get("timed_out") or "interrupt" in result.get("error", "") else error_type(result.get("error"))
    return result


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


class DatabaseTools:
    def __init__(self, row, metadata, disabled=(), error_only=False, timeout=10):
        self.db_path = row["db_path"]  # model cannot choose a filesystem path or database
        self.metadata = metadata
        self.disabled = set(disabled)
        self.error_only = error_only
        self.timeout = timeout
        self.cache = {}

    def catalog(self):
        return "\n".join(t["name"] for t in self.metadata["tables"])

    def search_schema(self, query, top_k=5):
        tokens = Counter(words(query))
        records = self.metadata["columns"]
        docs = [Counter(words(json.dumps(r, ensure_ascii=False))) for r in records]
        frequencies = Counter(w for d in docs for w in d)
        avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
        scored = []
        for i, d in enumerate(docs):
            score = 0.0
            for w in tokens:
                tf = d[w]
                if tf:
                    idf = math.log(1 + (len(docs) - frequencies[w] + .5) / (frequencies[w] + .5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * sum(d.values()) / max(avg, 1)))
            scored.append((score, i))
        selected = [records[i] for score, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:top_k] if score > 0]
        tables = {r["table"] for r in selected}
        return {"ok": True, "matches": selected,
                "table_schemas": [t for t in self.metadata["tables"] if t["name"] in tables]}

    def inspect_column(self, arguments):
        table, column = arguments["table"], arguments["column"]
        if not any(c["table"] == table and c["name"] == column for c in self.metadata["columns"]):
            return {"ok": False, "error": "Unknown table/column; use search_schema"}
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        # Bounded source sample: frequencies and ranges describe the sample, not the full DB.
        sql = f"SELECT {quote(column)} FROM {quote(table)} LIMIT 10000"
        result = run_sql(self.db_path, sql, limit=5, timeout=self.timeout)
        # preview is deliberately bounded. Lookup uses a bound parameter-free quoted literal.
        keyword = arguments.get("keyword", "")
        if keyword:
            literal = "'" + keyword.replace("'", "''").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "'"
            sql = (f"SELECT DISTINCT {quote(column)} FROM ({sql}) WHERE CAST({quote(column)} AS TEXT) "
                   f"LIKE '%' || {literal} || '%' ESCAPE '\\' LIMIT 6")
            result = run_sql(self.db_path, sql, limit=5, timeout=self.timeout)
        else:
            profile = run_sql(self.db_path,
                f"SELECT MIN({quote(column)}), MAX({quote(column)}), SUM({quote(column)} IS NULL), COUNT(*) FROM ({sql})",
                limit=1, timeout=self.timeout)
            if profile["ok"]:
                result["sample_min_max_nulls_count"] = profile["preview"][0]
        result.update({"sampled": True, "source_row_limit": 10000,
                       "note": "A missing value in this bounded sample does not prove absence."})
        return result

    def dispatch(self, action):
        if action.name in self.disabled:
            return {"ok": False, "error": "Tool unavailable in this ablation"}
        key = (action.name, json.dumps(action.arguments, sort_keys=True))
        if key in self.cache:
            return {**self.cache[key], "cached": True, "note": "Repeated call; budget is still consumed"}
        if action.name == "search_schema":
            result = self.search_schema(action.arguments)
        elif action.name == "inspect_column":
            result = self.inspect_column(action.arguments)
        elif action.name == "execute_sql":
            result = run_sql(self.db_path, action.arguments, timeout=self.timeout)
            if self.error_only and result["ok"]:
                result = {"ok": True}
        else:
            raise ValueError("not a tool action")
        self.cache[key] = result
        return result
