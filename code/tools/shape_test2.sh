#!/bin/bash
# Corrected collision geometry (16-gon stake prisms, 12-gon post prisms): pass-through check and teacher performance,
# old (v6g10t) vs new (tf2 final) bank, nominal + randomised, 16 rollouts per clip, criterion v2.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 \
  HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
O=$H/runs/shape_test2
mkdir -p $O
OLD=$(cat $H/runs/night/v6g10t_teachers.txt); NEW=$(cat $H/runs/tf2_final_teachers.txt)
for DRV in 0 1; do
  HNU_DR=$DRV CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $OLD --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O/dr$DRV --tag _old > $O/old$DRV.log 2>&1 &
  HNU_DR=$DRV CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $NEW --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O/dr$DRV --tag _new > $O/new$DRV.log 2>&1 &
  wait
done
tail -2 $O/new0.log
$PY - << 'EOF'
import json
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/shape_test2"
EXC = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split_v2.json"))["excluded"])
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
print("collision prisms; v2 success % DuMu / ThreeStepPoleRun / FiveStepPoleRun   (W1/W2 pass-through %)   [DuMu no-contact success %]")
for dr in ("0", "1"):
    for who in ("old", "new"):
        rows = [c for c in json.load(open(f"{O}/dr{dr}/eval_teacher_{who}.json"))["clips"] if c["clip"] not in EXC]
        g = lambda t, k: 100 * np.nanmean([c[k] for c in rows if c["task"] == t and c.get(k) is not None])  # noqa: E731
        print(f"{'nominal' if dr == '0' else 'random '} {who}: " + "/".join(f"{g(t, 'success_v2'):.0f}" for t in T) +
              "   (" + "/".join(f"{g(t, 'stake_pass_through'):.1f}" for t in T[1:]) + ")   [" + f"{g('CrouchedTraverse', 'success_no_contact'):.0f}]")
EOF
