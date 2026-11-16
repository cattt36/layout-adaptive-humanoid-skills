#!/bin/bash
# Diagnose the phase-BA drop: action-isolated (W2 only) Z0 latent on the new data, validated like the joint one.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
export PYTHONPATH=$H/data/data_code
OUT=$H/runs/diag_phase
mkdir -p $OUT
cd $H/code/stage4_latent
for P in Z0 Z4; do
  CUDA_VISIBLE_DEVICES=1 $PY motion_latent_z.py --mode train --datasets FiveStepPoleRun \
    --json-root $H/data/annot_v2/LLM_annotation --npz-root $H/data/latent_npz_v1 \
    --retargeted-json-root $H/data/annot_v2/Retargeted/Dynamics --ablation-profile $P --output-dir $OUT/$P \
    --epochs 200 --seed 42 > $OUT/train_$P.log 2>&1
  $PY validate_latents.py --latent-dir $OUT/$P --profile $P --json-root $H/data/annot_v2/LLM_annotation \
    --datasets FiveStepPoleRun > $OUT/validate_$P.log 2>&1
  grep -i -E 'phase.*balanced|balanced_accuracy' $OUT/validate_$P.log | head -3
  $PY - <<EOF
import json
r = json.load(open("$OUT/$P/latent_validation_report_$P.json"))
def find(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if "balanced" in k.lower() and isinstance(v, (int, float)):
                print("$P", path + "/" + k, round(v, 4))
            find(v, path + "/" + k)
find(r)
EOF
done
