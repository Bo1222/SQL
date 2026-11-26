#!/usr/bin/env bash
# P0-3 learning check: 4-question lr=5e-6 direction probe + 20-step GRPO.
#
# CPU-only preparation and GPU phases are separated. The 1-step checkpoint is
# exported, the direction probe must pass, then the 20-step run starts. The
# 1-step checkpoint is deleted after the probe and the 20-step FSDP shards are
# deleted after export because this host has ~21 GB free (single-checkpoint
# disk policy).
#
# Usage:
#   bash scripts/run_p0_3_learning_check.sh preflight
#   bash scripts/run_p0_3_learning_check.sh step1
#   bash scripts/run_p0_3_learning_check.sh direction
#   bash scripts/run_p0_3_learning_check.sh cleanup-step1
#   bash scripts/run_p0_3_learning_check.sh step20
#   bash scripts/run_p0_3_learning_check.sh cleanup-step20
#   bash scripts/run_p0_3_learning_check.sh after
#   bash scripts/run_p0_3_learning_check.sh all
#   bash scripts/run_p0_3_learning_check.sh status
#   bash scripts/run_p0_3_learning_check.sh watch        # refresh every 10 s
#   bash scripts/run_p0_3_learning_check.sh start        # detached `all`
#
# Detached start (recommended):
#   bash scripts/run_p0_3_learning_check.sh start
#   bash scripts/run_p0_3_learning_check.sh watch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPORT="reports/m6/rl_learning20_4q_5e6_20261119"
STEP1_OUT="outputs/m6_08m/rl_learning20_4q_5e6_step1_20261119"
STEP20_OUT="outputs/m6_08m/rl_learning20_4q_5e6_step20_20261119"
STEP1_RUN="rl_learning20_4q_5e6_step1_20261119"
STEP20_RUN="rl_learning20_4q_5e6_step20_20261119"
MODEL="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
TRAIN="reports/m6/rl_learning20_4q_20261118/train.parquet"
TASKS="reports/m6/rl_learning20_4q_20261118/tasks.jsonl"
VAL="artifacts/datasets/dev200_A_typed_v1.parquet"
BEFORE="reports/m6/rl_learning20_4q_20261118/sft_before/records.jsonl"
export WANDB_MODE=disabled CARDS=0 NPU_COUNT=1 CPU_THREADS=8
mkdir -p "$REPORT" "$REPORT/after32" logs/m6_08m_rl

log() { echo "[$(date '+%F %T')] $*" | tee -a "$REPORT/run.log"; }

write_status() {
  local phase="$1" error="${2:-}"
  local run_name="" out=""
  case "$phase" in
    step1|direction) run_name="$STEP1_RUN"; out="$STEP1_OUT" ;;
    step20|cleanup-step20|after) run_name="$STEP20_RUN"; out="$STEP20_OUT" ;;
  esac
  PHASE="$phase" ERROR="$error" RUN_NAME="$run_name" OUT="$out" \
  python3 - "$REPORT/status.json" <<'PY'
import json, os, sys, time
path = sys.argv[1]
try:
    data = json.loads(open(path).read()) if os.path.exists(path) else {}
except Exception:
    data = {}
data.update({"phase": os.environ.get("PHASE", "unknown"),
             "error": os.environ.get("ERROR") or None,
             "run_name": os.environ.get("RUN_NAME") or None,
             "out": os.environ.get("OUT") or None,
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
data.setdefault("started_utc", data["updated_utc"])
open(path, "w").write(json.dumps(data, indent=2) + "\n")
PY
}

fail() { write_status "failed" "$*"; log "ERROR: $*"; exit 1; }

trap 'write_status "failed" "line $LINENO: $BASH_COMMAND"' ERR

disk_free_gb() {
  df -Pk "$ROOT" | awk 'NR==2 {printf "%d", int($4/1024/1024)}'
}

require_disk() {
  local need="$1" free
  free="$(disk_free_gb)"
  log "disk free: ${free} GB (need >= ${need} GB)"
  if [ "$free" -lt "$need" ]; then
    fail "insufficient disk: ${free} GB free, need ${need} GB; do not start a checkpoint save"
  fi
}

checkpoint_ok() {
  local ckpt="$1"
  [ -f "$ckpt/config.json" ] || return 1
  find "$ckpt" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

run_launcher() {
  local phase="$1" steps="$2" save_freq="$3" run_name="$4" out="$5"
  log "launcher: $phase steps=$steps lr=5e-6 save_freq=$save_freq out=$out"
  RUN=1 \
  STEPS="$steps" EPOCHS=5 PROMPTS=1 MINI_BATCH=1 N=8 LR=5e-6 \
  CALCULATE_LOG_PROBS=1 SAVE_FREQ="$save_freq" MAX_CKPT_KEEP=1 \
  SAVE_CONTENTS=model,extra SAVE_OPTIMIZER=0 \
  REQUIRE_SFT_GATE=0 RUNTIME_CONTRACT=A-e1.2-typed-v1 \
  TRAIN="$TRAIN" VAL="$VAL" MODEL="$MODEL" OUT="$out" RUN_NAME="$run_name" \
  bash scripts/run_m6_08m_rl.sh 2>&1 | tee -a "$REPORT/${phase}_launcher.log"
}

export_checkpoint() {
  local out="$1" step="$2" ckpt="$1/hf_step_$2"
  if checkpoint_ok "$ckpt"; then
    log "export already present: $ckpt"
    return 0
  fi
  [ -d "$out/global_step_$step/actor" ] || fail "actor checkpoint missing: $out/global_step_$step/actor"
  log "exporting $out/global_step_$step/actor -> $ckpt"
  bash scripts/export_rl_checkpoint.sh "$out" "$step" 2>&1 | tee -a "$REPORT/export_step${step}.log"
  checkpoint_ok "$ckpt" || fail "export did not produce a loadable checkpoint: $ckpt"
  log "export ok: $ckpt"
}

preflight() {
  write_status "preflight"
  [ -f "$TRAIN" ] || fail "missing train parquet: $TRAIN"
  [ -f "$TASKS" ] || fail "missing tasks: $TASKS"
  [ -f "$BEFORE" ] || fail "missing before records: $BEFORE"
  [ -f "$MODEL/config.json" ] || fail "missing SFT model: $MODEL"
  [ -f "scripts/export_rl_checkpoint.sh" ] || fail "missing export script"
  python3 - "$REPORT/preflight.json" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    raise SystemExit("missing preflight.json; run the CPU preflight first")
data = json.loads(open(path).read())
required = [
    ("n_tasks", 4), ("n_before", 128), ("before_correct", 81),
    ("model_exists", True), ("prompt_fits", True),
]
for key, expected in required:
    if data.get(key) != expected:
        raise SystemExit(f"preflight {key}={data.get(key)!r}, expected {expected!r}")
if data.get("prompt_tokens_max", 10**9) > 2048:
    raise SystemExit("prompt token max exceeds 2048")
if not data.get("model_weight_files"):
    raise SystemExit("SFT model has no weight files")
print(json.dumps({"preflight": "ok", "prompt_tokens_max": data.get("prompt_tokens_max"),
                  "n_before": data.get("n_before"), "before_correct": data.get("before_correct"),
                  "disk_free_gb": data.get("disk_free_gb")}))
PY
  require_disk 20
  python3 scripts/probe_policy_update_direction.py --mock --require-direction \
    --out "$REPORT/mock_direction.json" >/dev/null
  python3 scripts/probe_policy_update_direction.py --mock-fail \
    --out "$REPORT/mock_direction_fail.json" >/dev/null 2>&1 || true
  log "CPU preflight and mock direction tests passed"
  write_status "preflight-ok"
}

step1() {
  write_status "step1"
  require_disk 20
  if checkpoint_ok "$STEP1_OUT/hf_step_1"; then
    log "1-step checkpoint already exported: $STEP1_OUT/hf_step_1"
    return 0
  fi
  run_launcher step1 1 1 "$STEP1_RUN" "$STEP1_OUT"
  export_checkpoint "$STEP1_OUT" 1
  write_status "step1-ok"
}

direction() {
  write_status "direction"
  checkpoint_ok "$STEP1_OUT/hf_step_1" || fail "1-step HF checkpoint missing"
  log "running direction probe (before=$MODEL after=$STEP1_OUT/hf_step_1)"
  python3 scripts/probe_policy_update_direction.py \
    --before-records "$BEFORE" \
    --before-model "$MODEL" \
    --after-model "$STEP1_OUT/hf_step_1" \
    --out "$REPORT/direction.json" \
    --device auto --require-direction 2>&1 | tee -a "$REPORT/direction.log"
  python3 - "$REPORT/direction.json" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
if not data.get("pass_direction"):
    raise SystemExit("direction gate failed")
print(json.dumps({key: data[key] for key in (
    "n", "n_correct", "n_incorrect", "delta_correct", "delta_incorrect",
    "gap_delta", "pass_direction")}, ensure_ascii=False))
PY
  write_status "direction-ok"
}

cleanup_step1() {
  case "$STEP1_OUT" in
    outputs/m6_08m/rl_learning20_4q_5e6_step1_20261119) ;;
    *) fail "refusing to delete unexpected path: $STEP1_OUT" ;;
  esac
  if [ -d "$STEP1_OUT" ]; then
    log "deleting 1-step checkpoint to free disk: $STEP1_OUT"
    rm -rf "$STEP1_OUT"
  fi
  write_status "step1-cleaned"
}

step20() {
  write_status "step20"
  require_disk 20
  if checkpoint_ok "$STEP20_OUT/hf_step_20"; then
    log "20-step checkpoint already exported: $STEP20_OUT/hf_step_20"
    return 0
  fi
  run_launcher step20 20 20 "$STEP20_RUN" "$STEP20_OUT"
  export_checkpoint "$STEP20_OUT" 20
  write_status "step20-ok"
}

cleanup_step20() {
  case "$STEP20_OUT" in
    outputs/m6_08m/rl_learning20_4q_5e6_step20_20261119) ;;
    *) fail "refusing to delete unexpected path: $STEP20_OUT" ;;
  esac
  if [ -d "$STEP20_OUT/global_step_20" ]; then
    log "deleting 20-step FSDP shards after export: $STEP20_OUT/global_step_20"
    rm -rf "$STEP20_OUT/global_step_20"
  fi
  write_status "step20-cleaned"
}

after() {
  write_status "after"
  checkpoint_ok "$STEP20_OUT/hf_step_20" || fail "20-step HF checkpoint missing"
  if [ -f "$REPORT/after32/summary.json" ] && \
     python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") else 1)' "$REPORT/after32/summary.json"; then
    log "after eval already complete: $REPORT/after32/summary.json"
    return 0
  fi
  log "running 32-sample after eval"
  python3 scripts/eval_rl_learning_pool.py \
    --model "$STEP20_OUT/hf_step_20" \
    --tasks "$TASKS" \
    --out "$REPORT/after32" \
    --seed 20261118 --samples 32 2>&1 | tee -a "$REPORT/after32.log"
  write_status "done"
  python3 - "$REPORT/after32/summary.json" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
print(json.dumps({key: data.get(key) for key in (
    "complete", "n_questions", "n_rollouts", "sampling_pass_at_1",
    "sampling_pass_at_32", "group_counts", "unavailable_rewards")}, ensure_ascii=False))
PY
}

status() {
  local run_name="$STEP1_RUN"
  if [ -f "$REPORT/status.json" ]; then
    run_name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name") or "")' "$REPORT/status.json" 2>/dev/null || true)"
    [ -n "$run_name" ] || run_name="$STEP1_RUN"
  fi
  python3 scripts/p0_3_status.py --report-dir "$REPORT" --run-name "$run_name" \
    --after-log "$REPORT/after32.log"
}

watch() {
  while true; do
    clear 2>/dev/null || true
    status
    echo "refresh: 10 s; Ctrl-C to stop"
    sleep 10
  done
}

case "${1:-status}" in
  preflight) preflight ;;
  step1) step1 ;;
  direction) direction ;;
  cleanup-step1) cleanup_step1 ;;
  step20) step20 ;;
  cleanup-step20) cleanup_step20 ;;
  after) after ;;
  all)
    preflight
    step1
    direction
    cleanup_step1
    step20
    cleanup_step20
    after
    write_status "done"
    log "P0-3 complete"
    ;;
  status) status ;;
  watch) watch ;;
  start)
    nohup bash "$0" all > "$REPORT/nohup.log" 2>&1 &
    echo $! > "$REPORT/runner.pid"
    log "started detached P0-3 runner pid=$(cat "$REPORT/runner.pid"); use 'watch' to monitor"
    ;;
  *) echo "usage: $0 {preflight|step1|direction|cleanup-step1|step20|cleanup-step20|after|all|start|status|watch}" >&2; exit 2 ;;
esac
