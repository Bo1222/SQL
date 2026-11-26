#!/usr/bin/env bash
# 4-prompt batch learning screen for the 4q pool (the reviewer-adjusted P0-3 step).
#
# screen : PROMPTS=4, MINI_BATCH=4, N=8, EPOCHS=5, STEPS=5 (5 optimizer steps,
#          each update sees all four questions; 32 rollouts/step).
# confirm: same batch config, STEPS=20, full gate.
#
# A separate enforcing monitor (scripts/p0_stop_monitor.py) kills the launcher
# on KL > 0.01, entropy < 0.01, grad outside (0, 10] for 3 steps, HBM > 30 GB,
# step > 12 min, unavailable/infra reward groups, or non-finite metrics.
#
# Usage:
#   bash scripts/run_p0_3_batch_check.sh preflight [screen|confirm]
#   bash scripts/run_p0_3_batch_check.sh all [screen|confirm]
#   bash scripts/run_p0_3_batch_check.sh status [screen|confirm]
#   bash scripts/run_p0_3_batch_check.sh watch [screen|confirm]
#   bash scripts/run_p0_3_batch_check.sh start [screen|confirm]   # detached all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${2:-screen}"
case "$MODE" in
  screen)  STEPS=5;  SAVE_FREQ=5;  EPOCHS=5;  SUFFIX=step5;  REPORT_SUFFIX=screen ;;
  confirm) STEPS=20; SAVE_FREQ=20; EPOCHS=20; SUFFIX=step20; REPORT_SUFFIX=confirm ;;
  *) echo "usage: $0 {preflight|all|status|watch|start} {screen|confirm}" >&2; exit 2 ;;
esac
export MODE STEPS EPOCHS LR RUN_TAG KL_COEF

RUN_TAG="${RUN_TAG:-}"
REPORT="reports/m6/rl_learning5batch_4q_5e6_${REPORT_SUFFIX}${RUN_TAG:+_$RUN_TAG}_20261120"
OUT="outputs/m6_08m/rl_learning5batch_4q_5e6_${SUFFIX}${RUN_TAG:+_$RUN_TAG}_20261120"
RUN_NAME="rl_learning5batch_4q_5e6_${SUFFIX}${RUN_TAG:+_$RUN_TAG}_20261120"
LOG="logs/m6_08m_rl/${RUN_NAME}.log"
TRAIN="reports/m6/rl_learning20_4q_20261118/train.parquet"
TASKS="reports/m6/rl_learning20_4q_20261118/tasks.jsonl"
VAL="artifacts/datasets/dev200_A_typed_v1.parquet"
BEFORE="reports/m6/rl_learning20_4q_20261118/sft_before/records.jsonl"
MODEL="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
PROMPTS=4
MINI_BATCH=4
N=8
LR="${LR:-5e-6}"
KL_COEF="${KL_COEF:-0.001}"
USE_KL_LOSS=True
DATA_SEED=42
export WANDB_MODE=disabled CARDS=0 NPU_COUNT=1 CPU_THREADS=8
mkdir -p "$REPORT" "$REPORT/after32" logs/m6_08m_rl

log() { echo "[$(date '+%F %T')] $*" | tee -a "$REPORT/run.log"; }

write_status() {
  local phase="$1" error="${2:-}" reason="${3:-}"
  PHASE="$phase" ERROR="$error" REASON="$reason" MODE="$MODE" RUN_NAME="$RUN_NAME" OUT="$OUT" \
  python3 - "$REPORT/status.json" <<'PY'
import json, os, sys, time
path = sys.argv[1]
try:
    data = json.loads(open(path).read()) if os.path.exists(path) else {}
except Exception:
    data = {}
data.update({"phase": os.environ.get("PHASE", "unknown"),
             "mode": os.environ.get("MODE", "unknown"),
             "error": os.environ.get("ERROR") or None,
             "stop_reason": os.environ.get("REASON") or None,
             "run_name": os.environ.get("RUN_NAME") or None,
             "out": os.environ.get("OUT") or None,
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
data.setdefault("started_utc", data["updated_utc"])
open(path, "w").write(json.dumps(data, indent=2) + "\n")
PY
}

fail() { write_status "failed" "$*" ""; log "ERROR: $*"; exit 1; }
trap 'write_status "failed" "line $LINENO: $BASH_COMMAND" ""' ERR

RAY_STARTED=0
cleanup_ray() {
  if [ "$RAY_STARTED" = "1" ]; then
    ( source scripts/env.sh; ray stop --force ) >/dev/null 2>&1 || true
    RAY_STARTED=0
  fi
}
trap cleanup_ray EXIT

disk_free_gb() { df -Pk "$ROOT" | awk 'NR==2 {printf "%d", int($4/1024/1024)}'; }

require_disk() {
  local need="${1:-20}" free
  free="$(disk_free_gb)"
  log "disk free: ${free} GB (need >= ${need} GB)"
  [ "$free" -ge "$need" ] || fail "insufficient disk: ${free} GB free, need ${need} GB"
}

checkpoint_ok() {
  local ckpt="$1"
  [ -f "$ckpt/config.json" ] || return 1
  find "$ckpt" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

preflight() {
  write_status "preflight"
  if [ -f "$REPORT/preflight.json" ]; then
    if python3 - "$REPORT/preflight.json" "$MODE" "$STEPS" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if (data.get("ok") and data.get("mode") == sys.argv[2]
                        and int(data.get("steps", -1)) == int(sys.argv[3])) else 1)
PY
    then
      [ ! -e "$OUT" ] || fail "output already exists: $OUT"
      require_disk 20
      log "reusing existing preflight.json (mode=$MODE steps=$STEPS)"
      write_status "preflight-ok"
      return 0
    fi
    fail "existing preflight.json does not match mode=$MODE steps=$STEPS"
  fi
  [ -f "$TRAIN" ] || fail "missing train parquet: $TRAIN"
  [ -f "$TASKS" ] || fail "missing tasks: $TASKS"
  [ -f "$BEFORE" ] || fail "missing before records: $BEFORE"
  [ -f "$MODEL/config.json" ] || fail "missing SFT model: $MODEL"
  [ -f "scripts/p0_stop_monitor.py" ] || fail "missing stop monitor"
  [ -f "scripts/export_rl_checkpoint.sh" ] || fail "missing export script"
  [ ! -e "$OUT" ] || fail "output already exists: $OUT (choose a fresh mode/path)"
  [ "$PROMPTS" -eq "$MINI_BATCH" ] || fail "PROMPTS=$PROMPTS must equal MINI_BATCH=$MINI_BATCH"
  [ $((PROMPTS * N)) -eq 32 ] || fail "PROMPTS*N must be 32 for this screen"
  require_disk 20
  python3 - "$REPORT/preflight.json" <<'PY'
import json, os, sys
path = sys.argv[1]
import math
try:
    import pyarrow.parquet as pq
    rows = pq.read_table('reports/m6/rl_learning20_4q_20261118/train.parquet').num_rows
except Exception:
    rows = None
before = sum(1 for _ in open('reports/m6/rl_learning20_4q_20261118/sft_before/records.jsonl'))
correct = 0
for line in open('reports/m6/rl_learning20_4q_20261118/sft_before/records.jsonl'):
    try:
        if json.loads(line).get('correct'):
            correct += 1
    except Exception:
        pass
data = {
    "mode": os.environ.get("MODE", "screen"),
    "rows": rows, "n_before": before, "before_correct": correct,
    "prompts": 4, "mini_batch": 4, "n": 8, "steps": int(os.environ.get("STEPS", "0")),
    "epochs": int(os.environ.get("EPOCHS", "0")),
    "lr": float(os.environ.get("LR", "5e-6")), "kl_coef": float(os.environ.get("KL_COEF", "0.001")), "use_kl_loss": True, "data_seed": 42,
    "effective_samples_per_update": 32,
    "disk_free_gb_df": int(os.popen("df -Pk . | awk 'NR==2{printf \"%d\", int($4/1024/1024)}'").read() or 0),
    "checks": {
        "train_rows_4": rows == 4,
        "before_128": before == 128,
        "before_correct_81": correct == 81,
        "prompts_mini_batch_equal": True,
        "effective_samples_32": True,
        "epochs_ge_steps": int(os.environ.get("EPOCHS", "0")) >= int(os.environ.get("STEPS", "0")),
    },
}
data["ok"] = all(data["checks"].values())
open(path, "w").write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(data, ensure_ascii=False))
if not data["ok"]:
    raise SystemExit("preflight checks failed")
PY
  write_status "preflight-ok"
  log "batch preflight passed"
}

run_training() {
  write_status "train"
  require_disk 20
  local launcher_pid monitor_pid rc
  rm -f "$REPORT/stop_trigger.json"
  : > "$LOG"
  log "launch: $MODE steps=$STEPS prompts=$PROMPTS mini_batch=$MINI_BATCH n=$N lr=$LR kl=$KL_COEF"
  RAY_STARTED=1
  setsid env RUN=1 WANDB_MODE=disabled CARDS=0 NPU_COUNT=1 CPU_THREADS=8 \
    STEPS="$STEPS" EPOCHS="$EPOCHS" PROMPTS="$PROMPTS" MINI_BATCH="$MINI_BATCH" N="$N" \
    LR="$LR" KL_COEF="$KL_COEF" USE_KL_LOSS="$USE_KL_LOSS" DATA_SEED="$DATA_SEED" \
    CALCULATE_LOG_PROBS=1 SAVE_FREQ="$SAVE_FREQ" MAX_CKPT_KEEP=1 \
    SAVE_CONTENTS=model,extra SAVE_OPTIMIZER=0 \
    REQUIRE_SFT_GATE=0 RUNTIME_CONTRACT=A-e1.2-typed-v1 \
    TRAIN="$TRAIN" VAL="$VAL" MODEL="$MODEL" OUT="$OUT" RUN_NAME="$RUN_NAME" \
    bash scripts/run_m6_08m_rl.sh > "$REPORT/train.log" 2>&1 &
  launcher_pid=$!
  echo "$launcher_pid" > "$REPORT/launcher.pid"
  log "launcher pid=$launcher_pid; starting enforcing stop monitor"
  python3 scripts/p0_stop_monitor.py \
    --log "$LOG" --pgid "$launcher_pid" --report-dir "$REPORT" \
    --interval 15 --step-seconds-max 900 > "$REPORT/monitor.log" 2>&1 &
  monitor_pid=$!
  echo "$monitor_pid" > "$REPORT/monitor.pid"
  sleep 1
  if ! kill -0 "$monitor_pid" 2>/dev/null; then
    kill -TERM -- -"$launcher_pid" 2>/dev/null || true
    cleanup_ray
    fail "stop monitor failed to start; launcher terminated"
  fi
  set +e
  wait "$launcher_pid"
  rc=$?
  set -e
  # Give the monitor one final poll to evaluate the last written step.
  for _ in $(seq 1 20); do
    kill -0 "$monitor_pid" 2>/dev/null || break
    sleep 1
  done
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  cleanup_ray
  if [ -f "$REPORT/stop_trigger.json" ]; then
    local reason
    reason="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason",""))' "$REPORT/stop_trigger.json")"
    write_status "stopped" "" "$reason"
    log "STOPPED by monitor: $reason"
    exit 2
  fi
  [ "$rc" -eq 0 ] || fail "launcher failed with rc=$rc"
  write_status "trained"
  log "training completed"
}

export_checkpoint() {
  local ckpt="$OUT/hf_step_$STEPS"
  if checkpoint_ok "$ckpt"; then
    log "export already present: $ckpt"
  else
    [ -d "$OUT/global_step_$STEPS/actor" ] || fail "actor checkpoint missing: $OUT/global_step_$STEPS/actor"
    log "exporting $OUT/global_step_$STEPS/actor -> $ckpt"
    bash scripts/export_rl_checkpoint.sh "$OUT" "$STEPS" 2>&1 | tee -a "$REPORT/export.log"
    checkpoint_ok "$ckpt" || fail "export did not produce a loadable checkpoint: $ckpt"
  fi
  case "$OUT" in
    outputs/m6_08m/rl_learning5batch_4q_5e6_*) ;;
    *) fail "refusing to delete unexpected path: $OUT" ;;
  esac
  if [ -d "$OUT/global_step_$STEPS" ]; then
    log "deleting FSDP shards after export: $OUT/global_step_$STEPS"
    rm -rf "$OUT/global_step_$STEPS"
  fi
  write_status "exported"
}

after_eval() {
  local ckpt="$OUT/hf_step_$STEPS"
  checkpoint_ok "$ckpt" || fail "missing HF checkpoint: $ckpt"
  if [ -f "$REPORT/after32/summary.json" ] && \
     python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") else 1)' "$REPORT/after32/summary.json"; then
    log "after eval already complete"
  else
    log "running 32-sample after eval"
    python3 scripts/eval_rl_learning_pool.py \
      --model "$ckpt" --tasks "$TASKS" --out "$REPORT/after32" \
      --seed 20261118 --samples 32 2>&1 | tee -a "$REPORT/after32.log"
  fi
  write_status "evaluated"
}

gate() {
  write_status "gate"
  python3 - "$REPORT" "$MODE" "$STEPS" "$BEFORE" "$LOG" <<'PY'
import json, sys, collections, re
report, mode, steps, before_path, log_path = sys.argv[1:6]
before = [json.loads(l) for l in open(before_path)]
after = [json.loads(l) for l in open(f'{report}/after32/records.jsonl')]
if len(before) != 128 or len(after) != 128:
    raise SystemExit(f"record count mismatch: before={len(before)} after={len(after)}")
before_n = collections.Counter(r['id'] for r in before)
after_n = collections.Counter(r['id'] for r in after)
records_32 = (len(before_n) == 4 and len(after_n) == 4
              and all(before_n[q] == 32 and after_n[q] == 32 for q in set(before_n) | set(after_n)))
after_summary = json.load(open(f'{report}/after32/summary.json'))
after_complete = bool(after_summary.get('complete'))
def counts(rows):
    c = collections.Counter()
    for r in rows:
        c[r['id']] += int(bool(r.get('correct')))
    return c
bc, ac = counts(before), counts(after)
qids = sorted(set(bc) | set(ac))
unavailable = sum(not r.get('reward_available', True) for r in after)
infra = sum(bool(r.get('infrastructure_error')) for r in after)
pass32 = all(ac[q] > 0 for q in qids)
no_collapse = all(ac[q] >= bc[q] - 3 for q in qids)
q4 = 'bird_train_713ccf94d1cb1e692f53'
q4_ok = ac.get(q4, 0) >= bc.get(q4, 0) - 3
before_total, after_total = sum(bc.values()), sum(ac.values())
manifest_keys = ['tasks_sha256', 'runtime_contract', 'runtime_fingerprint', 'seed',
                 'samples', 'temperature', 'top_p', 'top_k', 'prompt_ids_sha256', 'code_sha256']
bm = json.load(open('reports/m6/rl_learning20_4q_20261118/sft_before/manifest.json'))
am = json.load(open(f'{report}/after32/manifest.json'))
manifest_match = all(bm.get(k) == am.get(k) for k in manifest_keys)
# last-5 training metrics (confirm gate)
last_rewards, last_entropy, last_kl = [], [], []
if log_path and __import__('os').path.exists(log_path):
    for line in open(log_path, errors='replace'):
        if 'critic/score/mean' not in line:
            continue
        r = re.search(r'critic/score/mean:([0-9.eE+-]+)', line)
        e = re.search(r'actor/entropy:([0-9.eE+-]+)', line)
        k = re.search(r'actor/kl_loss:([0-9.eE+-]+)', line)
        if r: last_rewards.append(float(r.group(1)))
        if e: last_entropy.append(float(e.group(1)))
        if k: last_kl.append(float(k.group(1)))
last5_reward = sum(last_rewards[-5:]) / len(last_rewards[-5:]) if last_rewards else None
min_entropy = min(last_entropy) if last_entropy else None
max_kl = max(last_kl) if last_kl else None
logged_steps_ok = len(last_rewards) >= int(steps)
conditions = {
    'records_128_each': len(before) == 128 and len(after) == 128,
    'per_question_32_each': records_32,
    'after_summary_complete': after_complete,
    'same_condition_manifest': manifest_match,
    'logged_steps_present': logged_steps_ok,
    'unavailable_rewards_zero': unavailable == 0,
    'infrastructure_zero': infra == 0,
    'pass32_all_four': pass32,
    'q4_not_collapsed_geq_before_minus_3': q4_ok,
    'no_question_below_before_minus_3': no_collapse,
    'aggregate_not_worse_than_before': after_total >= before_total,
    'entropy_geq_0.01': min_entropy is not None and min_entropy >= 0.01,
    'kl_leq_0.01': max_kl is not None and max_kl <= 0.01,
}
if mode == 'confirm':
    conditions.update({
        'pass1_geq_96_of_128': after_total >= 96,
        'last5_reward_geq_0.95': last5_reward is not None and last5_reward >= 0.95,
    })
result = {
    'mode': mode, 'steps': int(steps),
    'before_total': before_total, 'after_total': after_total,
    'before_per_question': dict(bc), 'after_per_question': dict(ac),
    'pass1_before': before_total / len(before), 'pass1_after': after_total / len(after),
    'pass32': pass32, 'unavailable_rewards': unavailable, 'infrastructure_errors': infra,
    'last5_reward': last5_reward, 'min_entropy': min_entropy, 'max_kl': max_kl,
    'conditions': conditions, 'pass': all(conditions.values()),
}
open(f'{report}/gate.json', 'w').write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
lines = [f'# Batch {mode}: 4q, PROMPTS=4, STEPS={steps}', '',
         '| question | before | after |', '|---|---:|---:|']
for q in qids:
    lines.append(f'| {q} | {bc[q]}/32 | {ac[q]}/32 |')
lines += ['', f'pass@1: {before_total}/128 -> {after_total}/128',
          f'pass@32 all four: {pass32}',
          f'unavailable={unavailable}, infra={infra}',
          f'last5_reward={last5_reward}, min_entropy={min_entropy}, max_kl={max_kl}', '']
lines += [f'- {k}: {v}' for k, v in conditions.items()]
lines += ['', f'**PASS={result["pass"]}**', '']
open(f'{report}/RESULTS.md', 'w').write('\n'.join(lines))
print(json.dumps(result, ensure_ascii=False))
if not result['pass']:
    raise SystemExit(2)
PY
  write_status "done"
  log "batch gate PASS"
}

cleanup_stopped() {
  case "$OUT" in
    outputs/m6_08m/rl_learning5batch_4q_5e6_*) ;;
    *) fail "refusing to delete unexpected path: $OUT" ;;
  esac
  for path in "$OUT/global_step_$STEPS" "$OUT/hf_step_$STEPS"; do
    if [ -d "$path" ]; then
      log "deleting stopped-run checkpoint: $path"
      rm -rf "$path"
    fi
  done
  rm -f "$REPORT/stop_trigger.json"
  write_status "stopped-cleaned"
}

cleanup_screen() {
  local ckpt="outputs/m6_08m/rl_learning5batch_4q_5e6_step5_20261120/hf_step_5"
  case "$ckpt" in
    outputs/m6_08m/rl_learning5batch_4q_5e6_step5_20261120/hf_step_5) ;;
    *) fail "refusing to delete unexpected path: $ckpt" ;;
  esac
  if [ -d "$ckpt" ]; then
    log "deleting screen checkpoint after eval/gate: $ckpt"
    rm -rf "$ckpt"
  fi
  write_status "screen-cleaned"
}

status() {
  python3 scripts/p0_3_status.py --report-dir "$REPORT" --run-name "$RUN_NAME" \
    --after-log "$REPORT/after32.log" 2>/dev/null || true
  for f in monitor_status.json stop_trigger.json gate.json; do
    if [ -f "$REPORT/$f" ]; then echo "--- $f"; cat "$REPORT/$f"; fi
  done
}

watch() { while true; do clear 2>/dev/null || true; status; echo "refresh: 10 s; Ctrl-C to stop"; sleep 10; done; }

case "${1:-status}" in
  preflight) preflight ;;
  train) run_training ;;
  export) export_checkpoint ;;
  eval) after_eval ;;
  gate) gate ;;
  all) preflight; run_training; export_checkpoint; after_eval; gate ;;
  status) status ;;
  watch) watch ;;
  cleanup-screen) cleanup_screen ;;
  cleanup-stopped) cleanup_stopped ;;
  start)
    nohup bash "$0" all "$MODE" > "$REPORT/nohup.log" 2>&1 &
    echo $! > "$REPORT/runner.pid"
    log "started detached batch $MODE pid=$(cat "$REPORT/runner.pid"); use status/watch"
    ;;
  *) echo "usage: $0 {preflight|train|export|eval|gate|all|start|status|watch|cleanup-screen|cleanup-stopped} {screen|confirm}" >&2; exit 2 ;;
esac
