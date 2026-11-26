#!/usr/bin/env bash
# P0-1 (v3 prompt manipulation check) and P0-2 (forced exact-lookup probe).
#
# Every GPU phase is a separate command and checks its pre-registered gate
# before the next phase starts. No checkpoint is written.
#
# Usage:
#   bash scripts/run_p0_1_p0_2_v3.sh preflight
#   bash scripts/run_p0_1_p0_2_v3.sh smoke
#   bash scripts/run_p0_1_p0_2_v3.sh p01
#   bash scripts/run_p0_1_p0_2_v3.sh p02
#   bash scripts/run_p0_1_p0_2_v3.sh status
#   bash scripts/run_p0_1_p0_2_v3.sh all        # preflight -> smoke -> p01 -> p02
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPORT="reports/m6/harness_v3_20261120"
FORCED="$REPORT/forced_exact_dev200"
MODEL="outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
DATA="artifacts/datasets/dev200.jsonl"
DATA_SHA="bda04835e9404f872aafa892bc179ff80a9db1a11a946fc3e9c198b44c57a2ee"
CONTRACT="A-e1.2-typed-v3"
BASELINE="reports/m6/harness_audit_20261117/3b_dev200_harness_ab/records.jsonl"
export WANDB_MODE=disabled
mkdir -p "$REPORT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$REPORT/logs/runner.log"; }
fail() { log "ERROR: $*"; exit 1; }

disk_free_gb() {
  df -Pk "$ROOT" | awk 'NR==2 {printf "%d", int($4/1024/1024)}'
}

model_ok() {
  [ -f "$MODEL/config.json" ] || return 1
  find "$MODEL" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

preflight() {
  model_ok || fail "SFT model missing config or weights: $MODEL"
  [ -f "$DATA" ] || fail "missing dev200: $DATA"
  [ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || fail "dev200 sha256 mismatch"
  [ -s "$FORCED/seeds.jsonl" ] || fail "missing/empty seeds.jsonl; run prepare first"
  [ -s "$FORCED/prepare_summary.json" ] || fail "missing prepare_summary.json; run prepare first"
  python3 - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1])
prepare = json.loads((report / "forced_exact_dev200" / "prepare_summary.json").read_text())
assert prepare["n_seeded"] == prepare["n_rows"] == 200, prepare
readiness = json.loads((report / "readiness_v3.json").read_text())
assert readiness["ok"], readiness["errors"]
assert readiness["val"]["prompt_versions"] == ["catalog-three-action-query-v3"], readiness["val"]
assert readiness["val"]["retrieval_versions"] == ["E1.2"], readiness["val"]
print(json.dumps({"preflight": "ok", "forced_seeds": prepare["n_seeded"],
                  "readiness_ok": readiness["ok"]}))
PY
  free="$(disk_free_gb)"
  log "disk free: ${free} GB (inference-only; no checkpoint)"
  [ "$free" -ge 5 ] || fail "insufficient disk for inference logs: ${free} GB"
  log "P0-1/P0-2 preflight passed"
}

run_eval() {
  local limit="$1" out="$2" timeout_s="$3" tag="$4"
  log "P0-1 $tag: evaluator limit=$limit -> $out"
  timeout "$timeout_s" python3 eval/evaluate_agent.py \
    --data "$DATA" --limit "$limit" \
    --backend unified --engine vllm --model "$MODEL" \
    --prompt-mode query_v3 --retrieval-version E1.2 \
    --observation-mode compact_v2 --score-mode bird_set \
    --runtime-contract "$CONTRACT" \
    --out "$out" 2>&1 | tee -a "$REPORT/logs/p01_${tag}.log"
}

smoke() {
  preflight
  local out="$REPORT/manipulation_v3_e12_limit1.json"
  if [ -f "$out" ]; then
    log "smoke output already exists: $out"
  else
    run_eval 1 "$out" 1800 "limit1"
  fi
  python3 - "$out" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
records = [json.loads(line) for line in open(sys.argv[1].replace(".json", ".jsonl")) if line.strip()]
assert summary["prompt_version"] == "catalog-three-action-query-v3", summary["prompt_version"]
assert summary["prompt_mode"] == "query_v3", summary["prompt_mode"]
assert records and records[0]["prompt_version"] == "catalog-three-action-query-v3"
assert "catalog-three-action-query-v3" in records[0]["prompt_prefix"]
assert '<schema_lookup>{"table"' in records[0]["segments"][0]["text"]
print(json.dumps({"smoke": "ok", "n": summary["n"],
                  "prompt_version": summary["prompt_version"]}))
PY
  log "P0-1 limit-1 smoke passed"
}

p01() {
  preflight
  [ -f "$REPORT/manipulation_v3_e12_limit1.json" ] || fail "run the limit-1 smoke first"
  python3 - "$REPORT/manipulation_v3_e12_limit1.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["prompt_version"] == "catalog-three-action-query-v3", summary["prompt_version"]
print(json.dumps({"smoke_precondition": "ok", "n": summary["n"]}))
PY
  local out="$REPORT/manipulation_v3_e12.json"
  if [ -f "$out" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("n")==200 else 1)' "$out"; then
    log "P0-1 200 output already exists: $out"
  else
    run_eval 0 "$out" 5400 "200"
  fi
  python3 - "$out" "$MODEL" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["prompt_version"] == "catalog-three-action-query-v3", summary["prompt_version"]
assert summary["model"] == sys.argv[2], (summary["model"], sys.argv[2])
print(json.dumps({"p01_output": "ok", "model": summary["model"],
                  "prompt_version": summary["prompt_version"]}))
PY
  local usage="$REPORT/exact_usage_v3_e12.json"
  python3 scripts/analyze_exact_lookup_usage.py \
    --records "${out%.json}.jsonl" --out "$usage" \
    --markdown "$REPORT/exact_usage_v3_e12.md" >/dev/null
  python3 - "$out" "$usage" "$REPORT/p01_gate.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
records = [json.loads(line) for line in open(sys.argv[1].replace(".json", ".jsonl")) if line.strip()]
usage = json.load(open(sys.argv[2]))["aggregate"]["all"]
n = summary["n"]
checks = {
    "n_200": n == 200,
    "table_column_ge_40": usage["table_column"] >= 40,
    "exact_ok_ge_0.80": (usage["exact_ok_rate"] or 0) >= 0.80,
    "exact_inspection_ge_0.80": (usage["exact_inspection_rate"] or 0) >= 0.80,
    "exact_selected_ge_0.80": (usage["exact_selected_columns_rate"] or 0) >= 0.80,
    "ex_set_only_ge_120": round(summary["ex"] * n) >= 120,
    "valid_ge_169": round(summary["valid"] * n) >= 169,
    "invalid_le_2": sum(int(r.get("invalid_actions") or 0) for r in records) <= 2,
    "infra_le_2": summary["infra"] <= 2,
    "answer_ge_0.95": (1 - summary["no_answer"] / max(n, 1)) >= 0.95,
    "prompt_v3": summary["prompt_version"] == "catalog-three-action-query-v3",
}
gate = {"pass": all(checks.values()), "checks": checks,
        "table_column": usage["table_column"],
        "exact_questions": usage["exact_question_count"], "lookups": usage["lookups"],
        "exact_hit_rate": usage["exact_hit_rate"],
        "ex_set_only": round(summary["ex"] * n), "valid": round(summary["valid"] * n),
        "infra": summary["infra"], "prompt_version": summary["prompt_version"]}
open(sys.argv[3], "w").write(json.dumps(gate, indent=1) + "\n")
print(json.dumps(gate))
raise SystemExit(0 if gate["pass"] else 3)
PY
  log "P0-1 200 gate passed"
}

p02() {
  preflight
  [ -f "$FORCED/seeds.jsonl" ] || fail "forced seeds missing"
  local e12="$FORCED/records_E12_exact.jsonl" e11="$FORCED/records_E11_ignored.jsonl"
  if [ -f "$e12" ] && [ -f "$e11" ] && \
     [ "$(wc -l < "$e12")" -eq 200 ] && [ "$(wc -l < "$e11")" -eq 200 ]; then
    log "P0-2 records already complete"
  else
    log "P0-2 run: 400 episodes (E12_exact + E11_ignored)"
    timeout 7200 python3 scripts/probe_forced_exact_lookup.py run \
      --out "$FORCED" --model "$MODEL" \
      --arms E12_exact,E11_ignored --prompt-mode query_v3 --seed-lookup-from-gold \
      2>&1 | tee -a "$REPORT/logs/p02_run.log"
  fi
  python3 scripts/probe_forced_exact_lookup.py summarize \
    --out "$FORCED" --baseline-records "$BASELINE" \
    2>&1 | tee -a "$REPORT/logs/p02_summarize.log"
  python3 - "$FORCED/summary.json" "$REPORT/p02_gate.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
gate = summary["gate"]
if summary.get("prompt_versions") != ["catalog-three-action-query-v3"]:
    gate = {**gate, "pass": False, "prompt_versions": summary.get("prompt_versions")}
gate["n_pairs"] = summary["n_pairs"]
gate["ex"] = summary["ex"]
gate["discordant"] = summary["discordant"]
gate["sql_diff_rate"] = summary["sql_diff_rate"]
open(sys.argv[2], "w").write(json.dumps(gate, indent=1) + "\n")
print(json.dumps(gate))
raise SystemExit(0 if gate["pass"] else 3)
PY
  log "P0-2 gate passed"
}

status() {
  python3 - "$REPORT" "$FORCED" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1])
forced = Path(sys.argv[2])
def summary(path):
    return json.loads(path.read_text()) if path.is_file() else None
def count(path):
    return sum(1 for line in path.open() if line.strip()) if path.is_file() else 0
print("== P0-1/P0-2 status ==")
for tag, name in (("smoke", "manipulation_v3_e12_limit1.json"),
                  ("200", "manipulation_v3_e12.json")):
    data = summary(report / name)
    records = count(report / name.replace(".json", ".jsonl"))
    if data:
        print(f"P0-1 {tag}: n={data.get('n')} ex={data.get('ex')} valid={data.get('valid')} "
              f"infra={data.get('infra')} prompt={data.get('prompt_version')} records={records}")
    else:
        print(f"P0-1 {tag}: pending (records={records}/200)")
for arm in ("E12_exact", "E11_ignored"):
    print(f"P0-2 {arm}: records={count(forced / f'records_{arm}.jsonl')}/200")
for path in (report / "p01_gate.json", report / "p02_gate.json", forced / "summary.json"):
    data = summary(path)
    if data:
        gate = data.get("gate", data)
        print(f"{path.name}: pass={gate.get('pass')} checks={gate.get('checks')}")
print("logs:", ", ".join(sorted(p.name for p in (report / "logs").glob("*.log"))) if (report / "logs").is_dir() else "none")
PY
}

case "${1:-status}" in
  preflight) preflight ;;
  smoke) smoke ;;
  p01) p01 ;;
  p02) p02 ;;
  status) status ;;
  all)
    preflight
    smoke
    p01
    p02
    log "P0-1 and P0-2 complete"
    ;;
  *) echo "usage: $0 {preflight|smoke|p01|p02|status|all}" >&2; exit 2 ;;
esac
