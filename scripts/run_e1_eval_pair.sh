#!/usr/bin/env bash
# E1-06: merge adapter V2 and run the fixed dev200 control/treatment pair.
# Both arms use the frozen E1-01 contract (compact, 6144 context, greedy).
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
export PYTHONPATH=/home/ma-user/work/sqlrl:$PYTHONPATH
mkdir -p reports/m6/e1_alignment logs

ADAPTER=${ADAPTER:-outputs/e1_alignment/e1_alignment_adapter_v2}
MERGED=${MERGED:-outputs/e1_alignment/e1_alignment_adapter_v2_merged}

if [ ! -f "$MERGED/config.json" ]; then
  python3 scripts/merge_adapter.py \
    --base artifacts/checkpoints/step_600_merged \
    --adapter "$ADAPTER" --out "$MERGED"
fi

COMMON=(--data artifacts/datasets/dev200.jsonl --limit 0 --backend unified --engine vllm \
  --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024 \
  --observation-mode compact --max-context 6144 --max-model-len 6144)

if [ "${REUSE_CONTROL:-0}" != "1" ] || [ ! -f reports/m6/e1_alignment/dev200_sft.json ]; then
  python3 -u eval/evaluate_agent.py "${COMMON[@]}" \
    --model artifacts/checkpoints/step_600_merged \
    --out reports/m6/e1_alignment/dev200_sft.json
fi
cp reports/m6/e1_alignment/dev200_sft.jsonl reports/m6/e1_alignment/diag_sft_SFT.jsonl

python3 -u eval/evaluate_agent.py "${COMMON[@]}" \
  --model "$MERGED" \
  --out reports/m6/e1_alignment/dev200_v2.json
cp reports/m6/e1_alignment/dev200_v2.jsonl reports/m6/e1_alignment/diag_sft_V2.jsonl

python3 eval/paired_statistics.py \
  --a reports/m6/e1_alignment/dev200_sft.jsonl \
  --b reports/m6/e1_alignment/dev200_v2.jsonl \
  --out reports/m6/e1_alignment/dev200_paired.json

python3 -u -m e1_alignment.assess \
  --sft reports/m6/e1_alignment/dev200_sft.jsonl \
  --v2 reports/m6/e1_alignment/dev200_v2.jsonl \
  --out reports/m6/e1_alignment/assessment.json

python3 -u -m e1_alignment.make_report \
  --manifest artifacts/m6/e1_alignment/dataset_manifest.json \
  --train outputs/e1_alignment/e1_alignment_adapter_v2/train_summary.json \
  --sft reports/m6/e1_alignment/dev200_sft.json \
  --v2 reports/m6/e1_alignment/dev200_v2.json \
  --assessment reports/m6/e1_alignment/assessment.json \
  --out reports/m6/e1_alignment/assessment.md
