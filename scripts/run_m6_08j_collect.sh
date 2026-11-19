#!/usr/bin/env bash
set -eo pipefail
cd /home/ma-user/work/sqlrl
source scripts/env.sh
export PYTHONPATH=/home/ma-user/work/sqlrl:${PYTHONPATH:-}
python -m e1_alignment.rebuild.collect --source "$1"
