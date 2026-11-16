#!/bin/bash
# Stake collision shape test: does a 16-gon convex prism (same radius) or an equal-area box stop the feet from falling
# through the stakes in MuJoCo-Warp?  New teacher bank, nominal, 16 rollouts per clip; pass-through rate + v2 success.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0
O=$H/runs/shape_test
mkdir -p $O
CK=$(cat $H/runs/tf2_final_teachers.txt)
HNU_STAKE_SHAPE=mesh16 CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O --tag _mesh16 > $O/mesh16.log 2>&1 &
HNU_STAKE_SHAPE=box CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O --tag _box > $O/box.log 2>&1 &
wait
tail -3 $O/mesh16.log
$PY - << 'EOF'
import json, os
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/shape_test"
EXC = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split_v2.json"))["excluded"])
for tag, f in (("cylinder (current)", "/root/gpufree-data/hnu_v2/runs/pass_through/eval_teacher_new.json"),
               ("16-gon prism", f"{O}/eval_teacher_mesh16.json"), ("box equal area", f"{O}/eval_teacher_box.json")):
    if not os.path.exists(f):
        print(tag, "missing"); continue
    rows = [c for c in json.load(open(f))["clips"] if c["clip"] not in EXC]
    cells = []
    for t in ("ThreeStepPoleRun", "FiveStepPoleRun"):
        r = [c for c in rows if c["task"] == t]
        cells.append(f"{t}: pass-through {100 * np.mean([c['stake_pass_through'] for c in r]):5.1f} %, "
                     f"v2 {100 * np.mean([c['success_v2'] for c in r]):.0f} %, old {100 * np.mean([c['success'] for c in r]):.0f} %")
    print(f"{tag:20s} " + " | ".join(cells))
EOF
