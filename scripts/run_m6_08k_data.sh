#!/usr/bin/env bash
set -euo pipefail
cd /home/ma-user/work/sqlrl
if [[ "${1:-}" == sample ]]; then
  # Ascend's vendor setup probes unset shell variables (including ZSH_VERSION).
  set +u
  source scripts/env.sh
  set -u
fi
export PYTHONPATH="/home/ma-user/work/sqlrl/.codex-deps/m6-08k:/home/ma-user/work/sqlrl:${PYTHONPATH:-}"
mkdir -p reports/m6/sft_data_revision
exec /home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python -c '
import json, subprocess, sys, time
started = time.monotonic()
result = subprocess.run([sys.executable, "-m", "e1_alignment.revision.workflow", *sys.argv[1:]])
with open("reports/m6/sft_data_revision/phase_wall_times.jsonl", "a") as stream:
    stream.write(json.dumps({"phase": sys.argv[1:], "measured_wall_s": time.monotonic()-started, "exit": result.returncode}) + "\n")
raise SystemExit(result.returncode)
' "$@"
