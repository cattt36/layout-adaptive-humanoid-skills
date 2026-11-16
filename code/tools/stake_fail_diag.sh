#!/bin/bash
# Why do the stake experts fail?  Termination causes + failure time vs first-stake time (run-up references, 0.06 m).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=${AXY:-0.40} HNU_STAKE_SKIP_FIRST=0
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_r45_h06; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_r45_h06
  CK=$(cat $H/runs/stakes_eval/runup_h06/$T/checkpoint.txt)
  O=/tmp/faildiag_${AXY:-0.40}/$T; mkdir -p $O
  CUDA_VISIBLE_DEVICES=1 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --motion-dir $D/$T \
    --stake-top 0.06 --out $O > $O/log.txt 2>&1
  $PY - << EOF
import json, glob, numpy as np
j = json.load(open(glob.glob("$O/*.json")[0]))
print("$T causes:", j["termination_causes"])
fr = [x for c in j["clips"] for x in c["fail_frac_of_clip"]]
first = []
for f in glob.glob("$D/$T/*_stakes.json"):
    m = json.load(open(f)); n = len(np.load(f.replace("_stakes.json", "_motion.npz"))["joint_pos"])
    first.append(m["stakes"][0]["stance"][0] / m["fps"] * 50 / n)
print("$T fail at fraction of clip: median %.2f  quartiles %.2f-%.2f ; first stake at median %.2f" %
      (np.median(fr), np.percentile(fr, 25), np.percentile(fr, 75), np.median(first)))
EOF
done
