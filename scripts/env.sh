#!/usr/bin/env bash
# Shared environment for every model-facing job (CANN/ATB/HCCL, vLLM-Ascend spawn mode).
#
# Override SQLRL_VERL_ENV when the veRL quickstart environment lives elsewhere:
#   SQLRL_VERL_ENV=/path/to/verl_quickstart/scripts/env.sh source scripts/env.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SQLRL_VERL_ENV="${SQLRL_VERL_ENV:-$HOME/work/verl_quickstart/scripts/env.sh}"
if [ ! -f "$SQLRL_VERL_ENV" ]; then
  echo "error: veRL environment file not found: $SQLRL_VERL_ENV" >&2
  echo "       set SQLRL_VERL_ENV to the correct env.sh" >&2
  exit 1
fi
source "$SQLRL_VERL_ENV"
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_NPU_ALLOC_CONF
cd "$ROOT"
