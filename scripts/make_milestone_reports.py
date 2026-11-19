"""Consolidate milestone evidence from artifacts into reports/m1..m4.

Reads only files that earlier stages produced; never recomputes model outputs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "artifacts" / "manifests"


def load(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, payload):
    out = REPORTS / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def m1():
    splits = load(REPORTS / "split_audit.json") or load(MANIFESTS / "split_audit.json")
    validation = load(MANIFESTS / "dataset_validation.json")
    sft = load(MANIFESTS / "sft_dataset_summary.json")
    write("m1/data_engineering.json", {
        "milestone": "M1",
        "status": "pass" if validation and validation.get("status") == "pass" else "review",
        "splits": (splits or {}).get("splits"),
        "leakage_checks": (splits or {}).get("leakage_checks"),
        "dataset_validation": validation,
        "sft_dataset": sft,
        "deliverables": ["artifacts/catalog/*", "artifacts/datasets/rl_full.parquet",
                         "artifacts/datasets/sft_trajectories.jsonl",
                         "artifacts/manifests/split_audit.json",
                         "artifacts/manifests/dataset_validation.json"],
    })


def m2():
    catalog = load(MANIFESTS / "catalog_manifest.json")
    retrieval = load(REPORTS / "m2" / "retrieval_comparison_120.json")
    summary = (retrieval or {}).get("summary")
    write("m2/retrieval_backend.json", {
        "milestone": "M2",
        "status": "pass" if catalog and summary else "incomplete",
        "catalog": {k: (catalog or {}).get(k) for k in
                    ("stores", "tables", "columns", "relationships", "descriptions_matched",
                     "catalog_match_rate", "missing_databases")},
        "retrieval_comparison_120": summary,
        "deliverables": ["retrieval/{schema_extractor,catalog_parser,catalog_store,retriever,observation_formatter}.py",
                         "artifacts/catalog/<db_id>/{schema,catalog,relationships,retrieval_index,manifest}.json",
                         "reports/m2/retrieval_comparison_120.json"],
    })


def m3():
    smoke = load(REPORTS / "m4" / "smoke_val6.json")
    trace = REPORTS / "m4" / "traces_smoke2" / "episodes.jsonl"
    episodes = []
    if trace.is_file():
        episodes = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    finishes = {}
    for record in episodes:
        finishes[record.get("finish_reason")] = finishes.get(record.get("finish_reason"), 0) + 1
    write("m3/agent_loop.json", {
        "milestone": "M3",
        "status": "pass" if episodes else "incomplete",
        "episodes": len(episodes),
        "finish_reasons": finishes,
        "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in episodes) / max(len(episodes), 1), 2),
        "avg_schema_lookups": round(sum(r.get("schema_lookups", 0) for r in episodes) / max(len(episodes), 1), 2),
        "avg_sql_executes": round(sum(r.get("sql_executes", 0) for r in episodes) / max(len(episodes), 1), 2),
        "smoke_eval": {k: (smoke or {}).get(k) for k in
                       ("n", "ex", "valid", "infra", "avg_latency_ms", "avg_response_tokens")},
        "deliverables": ["agent/{protocol,sql_episode,bird_agent_loop}.py", "tools/{sql_worker,sql_execute,schema_lookup}.py",
                         "tests/test_protocol_tools_reward.py", "tests/test_retrieval.py",
                         "tests/test_dataset_contract.py"],
    })


def m4():
    smoke = load(REPORTS / "m4" / "smoke_val6.json")
    e2e = load(REPORTS / "m4" / "verl_e2e_step1.json")
    write("m4/reward_and_rollout.json", {
        "milestone": "M4",
        "status": "pass" if smoke and e2e else ("partial" if smoke else "incomplete"),
        "offline_rollout": smoke,
        "verl_agent_loop_step": e2e,
        "reward_checks": ["gold == candidate -> 1", "result-equivalent SQL -> 1",
                          "different result -> 0", "missing <answer> -> 0",
                          "duplicate rows change count -> 0"],
        "deliverables": ["reward/bird_ex_reward.py", "eval/{evaluate_agent,evaluate_retrieval,paired_statistics,gold_extraction,backends}.py",
                         "scripts/run_m4_e2e.sh", "configs/agent/agent_loops.yaml"],
    })



def m5():
    legacy = load(REPORTS / "m5" / "e_legacy_val100.json")
    unified = load(REPORTS / "m5" / "e_unified_val100.json")
    paired = load(REPORTS / "m5" / "e0_vs_e1_paired.json")
    write("m5/paired_retrieval_comparison.json", {
        "milestone": "M5",
        "status": "pass" if legacy and unified and paired else "incomplete",
        "frozen_adapter": "seed_0/step_600 (merged)", "rl_updates": 0,
        "e0_legacy": legacy, "e1_unified": unified, "paired": paired,
        "deliverables": ["reports/m5/e_legacy_val100.json", "reports/m5/e_unified_val100.json",
                         "reports/m5/e0_vs_e1_paired.json", "scripts/run_m5.sh"],
    })


def m6():
    cost = load(REPORTS / "m6" / "training_cost_estimate.json")
    write("m6/scale_up_plan.json", cost or {"status": "incomplete"})

if __name__ == "__main__":
    for name in sys.argv[1:] or ["m1", "m2", "m3", "m4", "m5", "m6"]:
        globals()[name]()

