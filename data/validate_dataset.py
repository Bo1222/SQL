"""Dataset quality audit for M1 (plan §4.7). Fails loudly on broken inputs."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward.bird_ex_reward import score_sql  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rl-parquet", default="artifacts/datasets/rl_full.parquet")
    p.add_argument("--sft-summary", default="artifacts/manifests/sft_dataset_summary.json")
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--out", default="artifacts/manifests/dataset_validation.json")
    a = p.parse_args()

    report = {"checks": {}, "failures": []}
    df = pd.read_parquet(a.rl_parquet)
    report["rl"] = {"rows": len(df), "dbs": df["extra_info"].map(lambda x: x["db_id"]).nunique()}
    keys = ("data_source", "prompt", "reward_model", "extra_info")
    report["checks"]["rl_columns"] = all(k in df.columns for k in keys)

    gold_ok = 0
    missing_db = 0
    gold_failures = []
    samples = df.sample(min(a.sample, len(df)), random_state=0)
    for _, row in samples.iterrows():
        extra = row["extra_info"]
        if not Path(extra["db_path"]).is_file():
            missing_db += 1
            continue
        gold = row["reward_model"]["ground_truth"]
        scored = {}
        for attempt in range(2):  # a heavy gold query may hit the 30s budget once
            scored = score_sql(extra["db_path"], gold, gold, timeout=30)
            if scored.get("correct"):
                break
        gold_ok += int(bool(scored.get("correct")))
        if not scored.get("correct"):
            gold_failures.append({"question_id": extra["question_id"], "db_id": extra["db_id"],
                                  "sql": gold[:200], "error": str(scored.get("error"))[:200],
                                  "error_type": scored.get("error_type"),
                                  "infrastructure": bool(scored.get("infrastructure_error")),
                                  "classification": "infrastructure" if scored.get("infrastructure_error")
                                  else "gold_query_failure"})
    report["rl"]["gold_executable_sample"] = f"{gold_ok}/{len(samples)}"
    report["rl"]["missing_db_sample"] = missing_db
    report["rl"]["gold_failures"] = gold_failures
    report["checks"]["gold_executable"] = gold_ok == len(samples) - missing_db
    report["checks"]["gold_failures_are_infrastructure"] = all(f["classification"] == "infrastructure" for f in gold_failures)
    report["checks"]["no_missing_databases"] = missing_db == 0

    # gold must never appear inside the prompt
    leaked = 0
    for _, row in samples.iterrows():
        prompt_text = "".join(m["content"] for m in row["prompt"])
        if row["reward_model"]["ground_truth"].strip() in prompt_text:
            leaked += 1
    report["checks"]["gold_not_in_prompt"] = leaked == 0
    report["rl"]["gold_in_prompt"] = leaked

    # DDL must not appear in the agent prompt
    ddl_like = df["prompt"].map(lambda msgs: "CREATE TABLE" in "".join(m["content"] for m in msgs).upper()).sum()
    report["checks"]["prompt_has_no_ddl"] = int(ddl_like) == 0
    report["rl"]["prompt_with_ddl"] = int(ddl_like)

    ids = df["extra_info"].map(lambda x: x["question_id"])
    report["checks"]["ids_unique"] = ids.is_unique
    report["rl"]["duplicate_ids"] = int(len(ids) - ids.nunique())

    if Path(a.sft_summary).is_file():
        sft = json.loads(Path(a.sft_summary).read_text(encoding="utf-8"))
        report["sft"] = {k: sft[k] for k in ("n", "dbs", "source_types", "final_ex_rate",
                                             "avg_tool_calls", "avg_masked_tokens")}
        report["checks"]["sft_all_final_correct"] = sft["final_ex_rate"] == 1.0

    report["failed_checks"] = [k for k, v in report["checks"].items() if not v]
    report["status"] = "pass" if not report["failed_checks"] else "fail"
    report["rl_parquet_sha256"] = hashlib.sha256(Path(a.rl_parquet).read_bytes()).hexdigest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failed": report["failed_checks"],
                      "rl": report["rl"], "sft": report.get("sft", {})}, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()

