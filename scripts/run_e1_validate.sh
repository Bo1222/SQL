#!/usr/bin/env bash
# E1-05: replay validation, dedupe, stratification and segment masks.
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
export PYTHONPATH=/home/ma-user/work/sqlrl:$PYTHONPATH

python3 -u -m e1_alignment.validate "$@"
