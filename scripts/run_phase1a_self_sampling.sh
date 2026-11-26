#!/usr/bin/env bash
# Phase 1A self-sampling on the 11 seen reviewed hard questions.
#
# The sampler has no early-stop hook, so it runs under an external monitor that
# kills its process group on the pre-registered stop rules. The 20 held-out
# questions are never sampled here.
#
# Usage:
#   bash scripts/run_phase1a_self_sampling.sh preflight
#   bash scripts/run_phase1a_self_sampling.sh run
#   bash scripts/run_phase1a_self_sampling.sh status
#   bash scripts/run_phase1a_self_sampling.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPORT="reports/m6/sft_first_20261120/self_sampling"
TASKS="$REPORT/tasks_11_seen.jsonl"
TASKS_SHA="11ad9edd35cef67d1a8352cfba87f0e00ffdf4b925e417027b1da87ca4ae2a47"
OUT="$REPORT/rollouts32"
EXPORT="$REPORT/export32"
MODEL="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
SEED=20261120
SAMPLES=32
LOG_DIR="$REPORT/logs"
SAMPLER_LOG="$LOG_DIR/phase1a_sampler.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$REPORT/phase1a_runner.log"; }
fail() { log "ERROR: $*"; exit 1; }

preflight() {
  [ -f "$TASKS" ] || fail "missing tasks: $TASKS"
  [ "$(sha256sum "$TASKS" | awk '{print $1}')" = "$TASKS_SHA" ] || fail "tasks sha256 mismatch"
  python3 - "$TASKS" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows) == 11, len(rows)
assert len({row["id"] for row in rows}) == 11
assert all(row.get("split") == "rl" for row in rows)
print(json.dumps({"tasks": len(rows), "dbs": len({row["db_id"] for row in rows})}))
PY
  [ -f "$MODEL/config.json" ] || fail "missing model config"
  find "$MODEL" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q . \
    || fail "model has no weight file"
  if [ "${RESUME:-0}" = "1" ]; then
    [ -f "$OUT/records.jsonl" ] && [ -f "$OUT/manifest.json" ] \
      || fail "RESUME=1 but records/manifest missing"
    [ ! -f "$REPORT/stop_trigger.json" ] && [ ! -f "$REPORT/low_yield.json" ] \
      || fail "previous stop/low-yield exists; use a fresh OUT"
    python3 - "$OUT/manifest.json" "$TASKS_SHA" "$SAMPLES" "$SEED" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest["tasks_sha256"] == sys.argv[2], "tasks hash mismatch"
assert manifest["samples"] == int(sys.argv[3]), "samples mismatch"
assert manifest["seed"] == int(sys.argv[4]), "seed mismatch"
assert manifest["runtime_contract"] == "A-e1.2-typed-v1", "contract mismatch"
print(json.dumps({"resume": "manifest ok"}))
PY
    python3 - "$OUT/records.jsonl" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows, bad = [], 0
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        bad += 1
path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
print(json.dumps({"resume_records": len(rows), "partial_dropped": bad}))
PY
  else
    [ ! -e "$OUT/records.jsonl" ] || fail "output exists; set RESUME=1 to resume"
  fi
  if ps -eo comm=,args= | awk '$1 ~ /^python/ && /eval_rl_learning_pool.py/ \
      && /tasks_11_seen/ {found=1} END {exit !found}'; then
    fail "a Phase 1A sampler is already running"
  fi
  python3 -m py_compile scripts/eval_rl_learning_pool.py scripts/phase1a_stop_monitor.py \
    scripts/export_self_sampled_sft.py
  free="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", int($4/1024/1024)}')"
  [ "$free" -ge 5 ] || fail "insufficient disk: ${free} GB"
  log "Phase 1A preflight passed (11 tasks, disk ${free} GB)"
}

run() {
  preflight
  mkdir -p "$OUT"
  cat > "$REPORT/run_manifest.json" <<EOF
{"phase":"1A_self_sampling","model":"$MODEL","tasks":"$TASKS","tasks_sha256":"$TASKS_SHA",
 "out":"$OUT","seed":$SEED,"samples":$SAMPLES,"temperature":1.0,"top_p":1.0,"top_k":-1,
 "contract":"A-e1.2-typed-v1","diagnostic_only":true,"started_utc":"$(date -u '+%Y-%m-%dT%H:%M:%SZ')"}
EOF
  log "starting sampler: 11 x $SAMPLES episodes"
  local pidfile="$OUT/.phase1a_sampler.pid"
  setsid --wait bash -c 'echo $$ > "$1"; exec env WANDB_MODE=offline PYTHONPATH="$2" \
    timeout 7200 python3 scripts/eval_rl_learning_pool.py --model "$3" --tasks "$4" \
    --out "$5" --seed "$6" --samples "$7"' _ "$pidfile" "$ROOT:${PYTHONPATH:-}" \
    "$MODEL" "$TASKS" "$OUT" "$SEED" "$SAMPLES" > "$SAMPLER_LOG" 2>&1 &
  local launcher_pid=$!
  for _ in $(seq 1 100); do [ -s "$pidfile" ] && break; sleep 0.1; done
  [ -s "$pidfile" ] || fail "sampler did not write a pid file; see $SAMPLER_LOG"
  local sampler_pid
  sampler_pid="$(cat "$pidfile")"
  kill -0 "$sampler_pid" 2>/dev/null || fail "sampler failed to start; see $SAMPLER_LOG"
  log "sampler pid=$sampler_pid pgid=$sampler_pid launcher=$launcher_pid; starting monitor"
  python3 scripts/phase1a_stop_monitor.py \
    --records "$OUT/records.jsonl" --pgid "$sampler_pid" --report-dir "$REPORT" \
    --interval 10 --seen-questions 8 --total-questions 11 --samples "$SAMPLES" \
    --min-correct-total 4 --stall-seconds 900 >> "$LOG_DIR/phase1a_monitor.log" 2>&1 &
  local monitor_pid=$!
  sleep 1
  kill -0 "$monitor_pid" 2>/dev/null || { kill -- -"$sampler_pid" 2>/dev/null || true; fail "monitor failed to start"; }
  set +e
  while kill -0 "$launcher_pid" 2>/dev/null; do
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
      kill -- -"$sampler_pid" 2>/dev/null || true
      set -e
      fail "monitor died while sampler was running; sampler killed"
    fi
    sleep 10
  done
  wait "$launcher_pid"; local sampler_rc=$?
  wait "$monitor_pid"; local monitor_rc=$?
  set -e
  log "sampler rc=$sampler_rc monitor rc=$monitor_rc"
  case "$monitor_rc" in
    0|2|3) ;;
    *) fail "monitor exited abnormally (rc=$monitor_rc)" ;;
  esac
  if [ -f "$REPORT/stop_trigger.json" ] && [ "$monitor_rc" -ne 2 ]; then
    fail "stop artifact exists but monitor rc=$monitor_rc (expected 2)"
  fi
  if [ "$monitor_rc" -eq 2 ] && [ ! -f "$REPORT/stop_trigger.json" ]; then
    fail "monitor rc=2 without stop_trigger.json"
  fi
  if [ -f "$REPORT/low_yield.json" ] && [ "$monitor_rc" -ne 3 ]; then
    fail "low-yield artifact exists but monitor rc=$monitor_rc (expected 3)"
  fi
  if [ "$monitor_rc" -eq 3 ] && [ ! -f "$REPORT/low_yield.json" ]; then
    fail "monitor rc=3 without low_yield.json"
  fi
  if [ "$sampler_rc" -ne 0 ] && [ ! -f "$REPORT/stop_trigger.json" ] \
     && [ ! -f "$REPORT/low_yield.json" ]; then
    fail "sampler exited unexpectedly (rc=$sampler_rc) without a stop/low-yield record"
  fi
  if [ -f "$REPORT/stop_trigger.json" ]; then
    cat "$REPORT/stop_trigger.json" | tee -a "$REPORT/phase1a_runner.log"
    fail "external monitor stopped the sampler; partial records preserved"
  fi
  if [ -f "$REPORT/low_yield.json" ]; then
    cat "$REPORT/low_yield.json" | tee -a "$REPORT/phase1a_runner.log"
    fail "low yield; no export/teacher/next stage"
  fi
  [ -f "$OUT/records.jsonl" ] || fail "sampler produced no records"
  log "sampler completed; exporting correct trajectories"
  python3 scripts/export_self_sampled_sft.py \
    --records "$OUT/records.jsonl" --manifest "$OUT/manifest.json" \
    --out "$EXPORT" --min-samples "$SAMPLES" --max-per-question 4 \
    --split-name hard_reviewed_11_seen 2>&1 | tee -a "$REPORT/phase1a_runner.log"
  python3 - "$EXPORT/sft_correct.jsonl" <<'PY'
import json, sys
forbidden = {"gold_sql", "gold_rows", "gold_error"}
def bad(value):
    if isinstance(value, dict):
        return bool(forbidden & set(value)) or any(bad(v) for v in value.values())
    if isinstance(value, list):
        return any(bad(v) for v in value)
    return False
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert rows, "no SFT rows"
assert not any(bad(row) for row in rows), "Gold-derived field in SFT export"
assert all(row["training_eligible"] and row["sft_split"] == "hard_reviewed_11_seen"
           for row in rows)
print(json.dumps({"sft_rows": len(rows), "forbidden_fields": "none"}))
PY
  log "Phase 1A complete: $EXPORT/sft_correct.jsonl"
}

status() {
  python3 - "$OUT" "$REPORT" "$EXPORT" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict
out, report, export = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
by = defaultdict(list)
p = out / "records.jsonl"
if p.is_file():
    for line in p.open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by[row["id"]].append(row)
print("== Phase 1A status ==")
print(f"records: {sum(len(v) for v in by.values())}/352; questions started: {len(by)}/11")
for qid, rows in sorted(by.items(), key=lambda kv: (-sum(r.get('correct', 0) for r in kv[1]), kv[0])):
    print(f"  {qid}: {sum(bool(r.get('correct')) for r in rows)}/{len(rows)} correct, "
          f"{sum(not r.get('reward_available', True) for r in rows)} unavailable")
for name in ("monitor_status.json", "stop_trigger.json", "low_yield.json"):
    path = report / name
    if path.is_file():
        print(name, path.read_text().strip().replace("\n", " "))
if (export / "yield_summary.json").is_file():
    summary = json.loads((export / "yield_summary.json").read_text())
    print("export:", {k: summary[k] for k in ("correct_trajectories_total",
                                               "all_zero_questions", "selected_for_sft")})
else:
    print("export: pending")
PY
}

case "${1:-status}" in
  preflight) preflight ;;
  run) run ;;
  status) status ;;
  all) preflight; run ;;
  *) echo "usage: $0 {preflight|run|status|all}" >&2; exit 2 ;;
esac
