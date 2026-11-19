"""Frozen copy of the previous `schema_lookup` implementation (E0 baseline).

Copied verbatim from the pre-restructure `agent/tools.py::DatabaseTools`
(schema_lookup / search_schema / catalog / words) so that E0 is byte-for-byte
the old retrieval logic, independent of the archived package.
"""
import json
import math
import re
from collections import Counter


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


class LegacySchemaLookupImpl:
    """E0: idf-weighted keyword search over column records, top-k -> tables."""

    def __init__(self, metadata):
        self.metadata = metadata

    def catalog(self):
        return "\n".join(t["name"] for t in self.metadata["tables"])

    def search_schema(self, query, top_k=5):
        tokens = Counter(words(query))
        records = self.metadata["columns"]

        def compact(record):
            return {
                "table": record["table"],
                "name": record["name"],
                "type": record.get("type", ""),
                "primary_key": bool(record.get("primary_key")),
                "description": (record.get("description") or "")[:180],
            }

        docs = [Counter(words(json.dumps(compact(r), ensure_ascii=False))) for r in records]
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
        selected = [compact(records[i]) for score, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:top_k] if score > 0]
        tables = list(dict.fromkeys(r["table"] for r in selected))
        return {"ok": True,
                "table_schemas": [next(t for t in self.metadata["tables"] if t["name"] == name) for name in tables],
                "matches": selected}

    def schema_lookup(self, arguments):
        table = (arguments.get("table") or "").strip()
        column = (arguments.get("column") or "").strip()
        query = (arguments.get("query") or "").strip()
        matches = []
        if table:
            selected = [t for t in self.metadata["tables"] if t["name"].lower() == table.lower()]
        elif query:
            search = self.search_schema(query)
            selected, matches = search.get("table_schemas", []), search.get("matches", [])
        else:
            selected = []
        if not selected:
            return {"status": "error", "error_type": "schema_error", "message": "unknown table or no matching table"}
        tables = []
        for entry in selected:
            columns = []
            for name in entry["columns"]:
                if column and name.lower() != column.lower():
                    continue
                details = next((c for c in self.metadata["columns"] if c["table"] == entry["name"] and c["name"] == name), {})
                columns.append({"name": name, "type": details.get("type", ""),
                                "primary_key": bool(details.get("primary_key")),
                                "description": details.get("description", "")})
            if columns:
                priority = [c["name"] for c in matches if c["table"] == entry["name"]]
                columns.sort(key=lambda c: priority.index(c["name"]) if c["name"] in priority else len(priority))
                tables.append({"name": entry["name"], "columns": columns,
                               "primary_keys": entry.get("primary_keys", []),
                               "foreign_keys": entry.get("foreign_keys", [])})
        if not tables:
            return {"status": "error", "error_type": "schema_error", "message": "unknown column"}
        return {"status": "ok", "tables": tables}
