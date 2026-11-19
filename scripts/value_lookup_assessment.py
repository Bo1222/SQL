"""Assess P0-#2a: value lookup vs SFT-B baseline on dev200."""
import json
import math
import re
from collections import Counter
from pathlib import Path


def load(path):
    return {r["id"]: r for r in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def literals(sql):
    return re.findall(r"'([^']*)'", sql or "")


def tables(sql):
    return set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", (sql or "").lower()))


def semantic_category(base):
    g, p = literals(base["gold_sql"]), literals(base["final_sql"])
    gs, ps = set(g), set(p)
    gl, pl = {x.lower() for x in g}, {x.lower() for x in p}
    if tables(base["gold_sql"]) != tables(base["final_sql"]):
        return "table_selection"
    if gs == ps:
        return "same_table_logic"
    if gl == pl:
        return "case_only_literal"
    return "literal_value_diff"


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    gain = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    loss = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    n = gain + loss
    k = min(gain, loss)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return {"n": len(ids), "gain": gain, "loss": loss, "net": gain - loss,
            "delta_ex": (gain - loss) / max(len(ids), 1), "p": p}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="reports/m6/diag_sft_B.jsonl")
    ap.add_argument("--value", default="reports/m6/value_lookup_ab.jsonl")
    ap.add_argument("--out", default="reports/m6/value_lookup_assessment.json")
    a = ap.parse_args()

    base, val = load(a.base), load(a.value)
    ids = sorted(set(base) & set(val))
    report = {"overall": {
        "base_ex": sum(base[i]["correct"] for i in ids) / len(ids),
        "value_ex": sum(val[i]["correct"] for i in ids) / len(ids),
        "base_valid": sum(base[i]["valid"] for i in ids) / len(ids),
        "value_valid": sum(val[i]["valid"] for i in ids) / len(ids),
    }, "mcnemar": mcnemar({i: base[i] for i in ids}, val), "semantic_fixes": {},
        "value_lookup_episodes": 0, "value_episodes_with_values": 0}

    for i in ids:
        if any((e.get("result") or {}).get("retrieval_metadata", {}).get("values")
               for e in val[i].get("tool_events", []) if e["action"] == "schema_lookup"):
            report["value_lookup_episodes"] += 1
        if any("Relevant Values:" in s.get("text", "") and "value like" in s.get("text", "")
               for s in val[i].get("segments", []) if s["kind"] == "observation"):
            report["value_episodes_with_values"] += 1

    for cat in ("case_only_literal", "literal_value_diff", "table_selection", "same_table_logic"):
        cases = [i for i in ids if base[i]["valid"] and not base[i]["correct"] and semantic_category(base[i]) == cat]
        fixed = [i for i in cases if val[i]["correct"]]
        report["semantic_fixes"][cat] = {"n": len(cases), "fixed": len(fixed), "ids": fixed}

    report["regressions"] = [
        {"id": i, "db": base[i]["db_id"], "base_error": base[i].get("error_type"),
         "value_error": val[i].get("error_type")}
        for i in ids if base[i]["correct"] and not val[i]["correct"]
    ]
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:5000])


if __name__ == "__main__":
    main()
