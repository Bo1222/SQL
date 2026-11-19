"""Assess the prompt A/B: case/literal + base-table guidance on dev200."""
import json
import math
import re
from collections import Counter
from pathlib import Path


def load(path):
    return {r["id"]: r for r in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def mcnemar(a, b):
    ids = sorted(set(a) & set(b))
    gain = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    loss = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    n = gain + loss
    k = min(gain, loss)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return {"n": len(ids), "gain": gain, "loss": loss, "net": gain - loss,
            "delta_ex": (gain - loss) / max(len(ids), 1), "mcnemar_p": p}


def literals(sql):
    return re.findall(r"'([^']*)'", sql or "")


def tables(sql):
    return set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", (sql or "").lower()))


def semantic_category(base):
    """Classify a valid-but-wrong B case by literal/table structure."""
    g, p = literals(base["gold_sql"]), literals(base["final_sql"])
    gs, ps = set(g), set(p)
    gl, pl = {x.lower() for x in g}, {x.lower() for x in p}
    tbl_same = tables(base["gold_sql"]) == tables(base["final_sql"])
    if not tbl_same:
        return "table_selection"
    if gs == ps:
        return "same_table_logic"
    if gl == pl:
        return "case_only_literal"
    return "literal_value_diff"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="reports/m6/diag_sft_B.jsonl")
    ap.add_argument("--prompt", default="reports/m6/prompt_ab_case_table.jsonl")
    ap.add_argument("--out", default="reports/m6/prompt_ab_assessment.json")
    a = ap.parse_args()

    base, prompt = load(a.base), load(a.prompt)
    ids = sorted(set(base) & set(prompt))
    report = {"overall": {
        "base_ex": sum(base[i]["correct"] for i in ids) / len(ids),
        "prompt_ex": sum(prompt[i]["correct"] for i in ids) / len(ids),
        "base_valid": sum(base[i]["valid"] for i in ids) / len(ids),
        "prompt_valid": sum(prompt[i]["valid"] for i in ids) / len(ids),
    }, "mcnemar": mcnemar({i: base[i] for i in ids}, prompt), "semantic_fixes": {}}

    for cat in ("table_selection", "same_table_logic", "case_only_literal", "literal_value_diff"):
        cases = [i for i in ids if base[i]["valid"] and not base[i]["correct"] and semantic_category(base[i]) == cat]
        fixed = [i for i in cases if prompt[i]["correct"]]
        report["semantic_fixes"][cat] = {"n": len(cases), "fixed": len(fixed),
                                          "ids": fixed[:10]}

    report["regressions"] = [
        {"id": i, "db": base[i]["db_id"], "base_error": base[i].get("error_type"),
         "prompt_error": prompt[i].get("error_type")}
        for i in ids if base[i]["correct"] and not prompt[i]["correct"]
    ]
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
