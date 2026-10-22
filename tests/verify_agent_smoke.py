"""Audit persisted smoke artifacts and actual W&B streams without rerunning training."""
import argparse
import json
import math
from pathlib import Path

from agent.common import read_jsonl, write_json


def verify_wandb(root):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    metadata = json.loads((root / "wandb_run.json").read_text())
    assert metadata["mode"] == "offline"
    stream = Path(metadata["run_dir"]) / f"run-{metadata['id']}.wandb"
    datastore = DataStore(); datastore.open_for_scan(str(stream))
    history, summary, exit_codes = [], {}, []
    def fields(items):
        return {(item.key or ".".join(item.nested_key)): json.loads(item.value_json) for item in items}
    try:
        while True:
            blob = datastore.scan_data()
            if blob is None:
                break
            record = wandb_internal_pb2.Record(); record.ParseFromString(blob)
            if record.HasField("history"): history.append(fields(record.history.item))
            if record.HasField("summary"): summary.update(fields(record.summary.update))
            if record.HasField("exit"): exit_codes.append(record.exit.exit_code)
    finally:
        datastore.close()
    local = read_jsonl(root / "train_log.jsonl")
    actual = [r for r in history if "grad_norm" in r]
    assert len(local) == len(actual) == 2
    for row, observed in zip(local, actual):
        for key, value in row.items():
            assert key in observed
            assert (math.isclose(value, observed[key], rel_tol=1e-8, abs_tol=1e-12)
                    if isinstance(value, float) else value == observed[key])
    assert summary["status"] == "completed" and exit_codes == [0]
    assert len(list((root / "wandb").glob("offline-run-*"))) == 1
    return {"id": metadata["id"], "mode": "offline", "matches_jsonl": True,
            "training_records": len(local), "status": summary["status"], "exit_codes": exit_codes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="reports/bird_agent_smoke_20261022")
    p.add_argument("--data", default="data/processed/bird_agent")
    a = p.parse_args()
    root, data = Path(a.root), Path(a.data)
    manifest = json.loads((data / "manifest.json").read_text())
    roles = {role: read_jsonl(data / f"{role}.jsonl") for role in ["sft", "rl", "val", "test"]}
    dbs = {role: {r["db_id"] for r in rows} for role, rows in roles.items()}
    assert not (dbs["sft"] | dbs["rl"]) & dbs["val"]
    assert not (dbs["sft"] | dbs["rl"] | dbs["val"]) & dbs["test"]
    sft_keys = {(r["db_id"], r["gold_sql"].strip()) for r in roles["sft"]}
    assert not sft_keys & {(r["db_id"], r["gold_sql"].strip()) for r in roles["rl"]}
    assert manifest["gold_validated"]
    rank_checks = [json.loads(path.read_text()) for path in sorted(root.glob("rank_*_checks.json"))]
    assert len(rank_checks) == 4
    assert all(r["reference_unchanged"] and r["policy_changed"] for r in rank_checks)
    training = json.loads((root / "grpo_npu/train_summary.json").read_text())
    assert training["effective_updates"] == training["optimizer_steps"] == 2
    assert training["world_size"] == 4 and training["requested_updates_reached"]
    long_path = root / "real_3b_long_context.json"
    if not long_path.is_file():
        long_path = root / "real_3b_long_context_2048_actions.json"
    long = json.loads(long_path.read_text())
    assert long["context_tokens"] == 6144 and long["reference_unchanged"]
    assert math.isfinite(long["grad_norm"]) and long["grad_norm"] > 0
    full_actions = json.loads((root / "real_3b_long_context_2048_actions.json").read_text())
    assert full_actions["context_tokens"] == 6144 and full_actions["action_tokens"] == 2048
    assert full_actions["reference_unchanged"] and math.isfinite(full_actions["grad_norm"])
    preflight = json.loads((root / "sql_sft_preflight.json").read_text())
    assert preflight["kept"] == len(roles["sft"]) and preflight["excluded"] == 0
    pilot = json.loads((root / "pilot.jsonl.summary.json").read_text())
    assert not pilot["ready_for_rl"]  # intentionally untrained fixture must fail the gate
    result = {"passed": True, "scope": "engineering smoke only; no full agent training or accuracy claim",
              "dataset": {"counts": {k: len(v) for k, v in roles.items()},
                          "annotation_revision": manifest["annotations_source"]["revision"],
                          "gold_validated": True, "database_isolation": True, "sql_group_isolation": True},
              "four_npu": {"effective_updates": 2, "rank_checks": rank_checks,
                           "best_checkpoint_saved": (root / "grpo_npu/best/adapter_config.json").is_file()},
              "grpo_wandb": verify_wandb(root / "grpo_npu"),
              "sft_wandb": verify_wandb(root / "tool_sft_cpu_tracked"),
              "sql_sft_preflight": preflight, "long_context": long,
              "long_context_full_action_budget": full_actions,
              "cold_start_gate_rejects_untrained_fixture": True,
              "native_3b_limited_eval": json.loads((root / "real_3b_native_eval.json").read_text())}
    write_json(root / "verification.json", result)
    print(json.dumps({"passed": True, "data": result["dataset"]["counts"],
                      "grpo_wandb": result["grpo_wandb"]["id"], "sft_wandb": result["sft_wandb"]["id"],
                      "context_tokens": full_actions["context_tokens"], "action_tokens": full_actions["action_tokens"],
                      "peak_mem_gb": full_actions["peak_mem_gb"]}))


if __name__ == "__main__":
    main()
