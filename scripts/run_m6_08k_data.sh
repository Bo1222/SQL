#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail
cd "$ROOT"
if [[ "${1:-}" == sample ]]; then
  # Ascend's vendor setup probes unset shell variables (including ZSH_VERSION).
  set +u
  source "$ROOT/scripts/env.sh"
  set -u
fi
export PYTHONPATH="$ROOT/.codex-deps/m6-08k:$ROOT:${PYTHONPATH:-}"
mkdir -p reports/m6/sft_data_revision
exec python3 -c '
import json, subprocess, sys, time
started = time.monotonic()
result = subprocess.run([sys.executable, "-m", "e1_alignment.revision.workflow", *sys.argv[1:]])
with open("reports/m6/sft_data_revision/phase_wall_times.jsonl", "a") as stream:
    stream.write(json.dumps({"phase": sys.argv[1:], "measured_wall_s": time.monotonic()-started, "exit": result.returncode}) + "\n")
raise SystemExit(result.returncode)
' "$@"
