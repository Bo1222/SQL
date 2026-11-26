#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -eo pipefail
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m e1_alignment.rebuild.collect --source "$1"
