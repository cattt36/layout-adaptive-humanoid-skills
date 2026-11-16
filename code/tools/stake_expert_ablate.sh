#!/bin/bash
# Isolate the early failure of the W1 stake expert (runup_h06):
#   A: its own data (0.06 m, stride 0.93, run-up warp), no drift/stake terminations
#   B: unscaled near-flat data (0.01 m, stride 1.0), no drift/stake terminations
#   C: flat expert on its own data A (same conditions)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
CK=$(cat $H/runs/stakes_eval/runup_h06/ThreeStepPoleRun/checkpoint.txt)
FL=$(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt)
run() {  # tag ckpt data height perception
  local O=/tmp/ablate/$1; mkdir -p $O
  HNU_PERCEPTION=$5 HNU_STAKE_H=$4 HNU_ANCHOR_XY=100 HNU_STAKE_MISS_TERM=0 HNU_STAKES_ROOT=$3 CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py \
    --task ThreeStepPoleRun --stakes --ckpt $2 --per-clip 8 --motion-dir $3/ThreeStepPoleRun --stake-top $4 --out $O > $O/log.txt 2>&1
  $PY - << EOF
import json, glob, numpy as np
j = json.load(open(glob.glob("$O/*.json")[0]))
fr = [x for c in j["clips"] for x in c["fail_frac_of_clip"]]
print("$1: completed", round(j["success_rate"], 3), "| all stakes hit", round(j["success_all_stakes"], 3), "| causes", j["termination_causes"],
      "| median fail frac", round(float(np.median(fr)), 2) if fr else None)
EOF
}
run A_stake_expert_own_data $CK $H/data/ref_stakes_s93_r45_h06 0.06 1
run B_stake_expert_flatdata $CK $H/data/ref_stakes_s93_h01 0.01 1
run C_flat_expert_own_data $FL $H/data/ref_stakes_s93_r45_h06 0.06 0
