"""One-line veRL patch: pass per-trajectory info (step/sample/rollout) into the agent loop.

Idempotent. Run on the server before M6 calibration.
"""
from pathlib import Path

TARGET = Path("/home/ma-user/verl/verl/experimental/agent_loop/agent_loop.py")
OLD = "            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}\n"
NEW = OLD + "            kwargs[\"_trajectory_info\"] = trajectory_info[i]\n"


def main():
    text = TARGET.read_text()
    if "kwargs[\"_trajectory_info\"]" in text:
        print("already patched")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"unexpected anchor count: {count}")
    TARGET.write_text(text.replace(OLD, NEW))
    print("patched")


if __name__ == "__main__":
    main()

