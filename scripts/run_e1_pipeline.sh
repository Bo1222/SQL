#!/usr/bin/env bash
# Full M6-08i orchestration. Each stage is independently resumable/rerunnable.
# GPU stages (A, B, control eval, SFT, V2 eval) must run sequentially on one NPU.
# C and validation are CPU-only but are kept in the same order for auditability.
set -eo pipefail
cd /home/ma-user/work/sqlrl

bash scripts/run_e1_collect_a.sh
bash scripts/run_e1_collect_b.sh
bash scripts/run_e1_collect_c.sh
bash scripts/run_e1_validate.sh
bash scripts/run_e1_sft.sh
bash scripts/run_e1_eval_pair.sh
