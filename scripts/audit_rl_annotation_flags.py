"""Execute evidence-literal/boundary alternatives for a manual training audit.

This writes review evidence only. It neither relabels data nor excludes tasks.
The scan is a narrow heuristic and cannot estimate total annotation noise.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reward.bird_ex_reward import score_sql


def evidence_alternative(sql, flags):
    replacements = []
    for flag in flags:
        if flag["kind"] == "literal_case_disagreement":
            old, new = flag["gold_value"], flag["evidence_value"]
            pattern = r"(\b" + re.escape(flag["column"]) + r"[`\"]?\s*=\s*)'" + re.escape(old) + "'"
            changed, count = re.subn(pattern, lambda m: m[1] + "'" + new + "'", sql, flags=re.I)
        else:
            pattern = (r"(\b" + re.escape(flag["column"]) + r"[`\"]?\s*)"
                       + re.escape(flag["gold_operator"]) + r"\s*"
                       + re.escape(flag["value"]) + r"(?![\d.])")
            changed, count = re.subn(pattern, lambda m: m[1] + flag["evidence_operator"] + flag["value"],
                                     sql, flags=re.I)
        if count:
            sql = changed
            replacements.append({"flag": flag, "occurrences": count})
    return sql, replacements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags", default="reports/m6/rl_signal_20261118/annotation_flags.jsonl")
    parser.add_argument("--out", default="reports/m6/rl_preflight_20261118/annotation_execution_review.jsonl")
    args = parser.parse_args()
    flags = [json.loads(s) for s in Path(args.flags).read_text().splitlines() if s.strip()]
    rows = {r['id']: r for r in map(json.loads, (ROOT / 'data/processed/bird_agent/rl.jsonl').read_text().splitlines())}
    out = Path(args.out)
    if out.exists():
        raise ValueError("choose a new review output")
    out.parent.mkdir(parents=True, exist_ok=True)
    for item in flags:
        row = rows[item['id']]
        alternative, replacements = evidence_alternative(row['gold_sql'], item['flags'])
        if not replacements:
            raise ValueError(f"no replacement for {row['id']}")
        result = score_sql(row['db_path'], alternative, row['gold_sql'], score_mode="bird_set", typed=True)
        receipt = {**item, "evidence_alternative_sql": alternative, "replacements": replacements,
                   "execution_comparison": result, "reviewed_label": None,
                   "disposition": "review_only_original_data_unchanged"}
        with out.open('a') as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False) + '\n')
        print(json.dumps({'id': row['id'], 'score_available': result['reward_available'],
                          'alternative_executes': result['ok'], 'same_result': result['correct']}), flush=True)


if __name__ == '__main__':
    main()
