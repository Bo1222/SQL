#!/usr/bin/env bash
# Shared environment for every model-facing job (CANN/ATB/HCCL, vLLM-Ascend spawn mode).
set -eo pipefail
source /home/ma-user/work/verl_quickstart/scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_NPU_ALLOC_CONF
cd /home/ma-user/work/sqlrl
