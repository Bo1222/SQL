#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# E1-05: replay validation, dedupe, stratification and segment masks.
set -eo pipefail
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

python3 -u -m e1_alignment.validate "$@"
