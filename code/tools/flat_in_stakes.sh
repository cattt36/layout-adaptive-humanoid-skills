#!/bin/bash
# Control experiment: the FLAT W1 expert (98 % on flat ground) evaluated in the stakes environment with stakes of
# height 0.01 m (practically flat), no drift / stake terminations.  If it also fails at ~20 % of the clip, the cause
# is the stakes environment set-up, not the stakes.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
O=/tmp/flat_in_stakes; mkdir -p $O
D=$H/data/ref_stakes_s93_h01
[ -d $D/ThreeStepPoleRun ] || $PY stage1_reference/make_stakes.py --task ThreeStepPoleRun --ref $H/data/ref_v1/ThreeStepPoleRun \
  --csv $H/data/scene/ThreeStepPoleRun.csv --height 0.01 --hscale 1.0 --out $D > /dev/null 2>&1
HNU_STAKE_H=0.01 HNU_ANCHOR_XY=100 HNU_STAKE_MISS_TERM=0 HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py \
  --task ThreeStepPoleRun --stakes --ckpt $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) --per-clip 8 --motion-dir $D/ThreeStepPoleRun \
  --stake-top 0.01 --out $O > $O/log.txt 2>&1
$PY - << 'EOF'
import json, glob, numpy as np
j = json.load(open(glob.glob("/tmp/flat_in_stakes/*.json")[0]))
fr = [x for c in j["clips"] for x in c["fail_frac_of_clip"]]
print("flat expert in stakes env: success", round(j["success_rate"], 3), "causes", j["termination_causes"],
      "median fail frac", np.median(fr) if fr else None)
EOF
