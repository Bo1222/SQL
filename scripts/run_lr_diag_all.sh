#!/usr/bin/env bash
# Run the three 5-step LR diagnostics sequentially on the single visible NPU.
set -o pipefail
cd /home/ma-user/work/sqlrl
mkdir -p logs artifacts/checkpoints/lr_diag reports/m6/traces_lr_diag

for spec in "0.000001 lr1e6" "0.000002 lr2e6" "0.000005 lr5e6"; do
  set -- $spec
  lr="$1"; tag="$2"
  echo "[$(date '+%F %T')] start $tag lr=$lr"
  bash scripts/run_lr_diag.sh "$lr" "$tag" 5 > "logs/lr_diag_${tag}.log" 2>&1
  echo "[$(date '+%F %T')] $tag exit=$?"
done
echo "done"
