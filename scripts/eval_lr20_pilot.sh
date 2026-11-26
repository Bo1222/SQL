#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Evaluate the 20-step 5e-6 pilot at steps 10 and 20 on dev200 (compact).
set -o pipefail
cd "$ROOT"
export CARDS=0
bash scripts/eval_lr_diag.sh lr5e6_20 10 > logs/eval_lr5e6_20_step10.log 2>&1
echo "step10 exit=$?"
bash scripts/eval_lr_diag.sh lr5e6_20 20 > logs/eval_lr5e6_20_step20.log 2>&1
echo "step20 exit=$?"
echo done
