"""M6-08i E1 tool-environment alignment SFT pipeline.

Sub-stages (docs/TASKS_M6.md):
  E1-01 fingerprint.py     fixed runtime contract
  E1-02 collect_a.py       E1 self rollouts from the original SFT policy
  E1-03 collect_b.py       continuation from real failed states
  E1-04 collect_c.py       gold-AST-assisted programmatic demonstrations
  E1-05 validate.py        replay validation, dedupe, stratification, masks
  E1-06 sft_train.py       segment-masked LoRA SFT V2 + dev200 comparison
"""
