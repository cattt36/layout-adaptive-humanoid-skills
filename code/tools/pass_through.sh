#!/bin/bash
# How often does a foot fall through a stake (MuJoCo-Warp sphere-cylinder contact loss)?  Teachers (current / new)
# and the v6g10m seed-1 L0 / L4 students, nominal, 16 rollouts per clip, all trials.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0
O=$H/runs/pass_through
mkdir -p $O
CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $(cat $H/runs/night/v6g10t_teachers.txt) --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O --tag _cur > $O/cur.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $(cat $H/runs/tf2_final_teachers.txt) --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O --tag _new > $O/new.log 2>&1 &
wait
for P in Z0 Z4; do
  CUDA_VISIBLE_DEVICES=$([ $P = Z0 ] && echo 0 || echo 1) $PY eval_student.py --student $H/runs/students_v6g10m_d2_s1/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5t/$P --per-clip 16 --out $O --tag _$P > $O/$P.log 2>&1 &
done
wait
$PY - << 'EOF'
import json
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/pass_through"
EXC = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split_v2.json"))["excluded"])
for f, name in (("eval_teacher_cur", "current teacher"), ("eval_teacher_new", "new teacher"), ("eval_student_Z0", "L0 m s1"), ("eval_student_Z4", "L4 m s1")):
    rows = [c for c in json.load(open(f"{O}/{f}.json"))["clips"] if c["clip"] not in EXC]
    cells = []
    for t in ("ThreeStepPoleRun", "FiveStepPoleRun"):
        r = [c for c in rows if c["task"] == t]
        pt = np.mean([c["stake_pass_through"] for c in r])
        v2 = np.mean([c["success_v2"] for c in r])
        cells.append(f"{t}: pass-through in {100 * pt:.1f} % of rollouts, v2 success {100 * v2:.0f} %")
    print(f"{name:16s} " + " | ".join(cells))
EOF
