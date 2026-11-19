#!/usr/bin/env bash
# Merge a veRL FSDP actor checkpoint into a HuggingFace model directory.
#
# Usage:
#   bash scripts/export_rl_checkpoint.sh RUN_DIR STEP [OUT_DIR]
# Example:
#   bash scripts/export_rl_checkpoint.sh outputs/m6_08m/rl_r1_grpo_20 10
#   -> outputs/m6_08m/rl_r1_grpo_20/hf_step_10
#
# Optional:
#   OVERWRITE=1   replace an existing target directory
#   DELETE_SHARDS=1 delete <run>/global_step_N/actor after a verified merge
set -eo pipefail
cd /home/ma-user/work/sqlrl
source scripts/env.sh
export PYTHONPATH=/home/ma-user/work/sqlrl:${PYTHONPATH:-}

RUN_DIR="${1:?usage: $0 RUN_DIR STEP [OUT_DIR]}"
STEP="${2:?usage: $0 RUN_DIR STEP [OUT_DIR]}"
ACTOR_DIR="${RUN_DIR}/global_step_${STEP}/actor"
OUT_DIR="${3:-${RUN_DIR}/hf_step_${STEP}}"

if [ ! -d "$ACTOR_DIR" ]; then
  echo "actor checkpoint missing: $ACTOR_DIR" >&2
  exit 1
fi
if [ -e "$OUT_DIR" ] && [ "${OVERWRITE:-0}" != "1" ]; then
  echo "target exists: $OUT_DIR (set OVERWRITE=1 to replace)" >&2
  exit 1
fi
if [ "${OVERWRITE:-0}" = "1" ]; then
  rm -rf "$OUT_DIR"
fi

mkdir -p "$OUT_DIR" logs/m6_08m_rl
LOG="logs/m6_08m_rl/export_$(basename "$RUN_DIR")_step${STEP}.log"
echo "merging $ACTOR_DIR -> $OUT_DIR" | tee -a "$LOG"
python -u -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = root / "config.json"
if not config.is_file():
    raise SystemExit(f"missing config.json in {root}")
cfg = json.loads(config.read_text(encoding="utf-8"))
weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
if not weights:
    raise SystemExit(f"no HF weight file in {root}")
print(json.dumps({
    "path": str(root),
    "architectures": cfg.get("architectures"),
    "dtype": cfg.get("dtype") or cfg.get("torch_dtype"),
    "weight_files": [str(p.name) for p in weights],
}, ensure_ascii=False, indent=1))
PY

if [ "${DELETE_SHARDS:-0}" = "1" ]; then
  echo "deleting FSDP shards after verified merge: $ACTOR_DIR" | tee -a "$LOG"
  rm -rf "$ACTOR_DIR"
fi
echo "export ok: $OUT_DIR" | tee -a "$LOG"
