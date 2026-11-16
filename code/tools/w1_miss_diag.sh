#!/bin/bash
# Which W1 stake is missed by the teachers, and how (closest loaded-foot landing relative to the stake centre)?
# Current vs new (w1b) W1 teacher, nominal, 32 rollouts per W1 clip (split-v2 exclusions removed).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0 \
  HNU_EVAL_DETAIL=1
O=$H/runs/w1_miss
mkdir -p $O
( ls $H/data/ref_stakes_s93_h10/ThreeStepPoleRun/*_motion.npz | xargs -n1 basename | sed 's/_motion.npz//' \
    | grep -v -e _022 -e _025 -e _029; echo CrouchedTraverse_008 ) > $O/clips.txt
export HNU_CLIPS_ALLOW=$O/clips.txt
CUR=$(cat $H/runs/night/v6g10t_teachers.txt)
NEW=$(cat $H/runs/tf2_final_teachers.txt)
CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CUR --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 --out $O --tag _current > $O/current.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $NEW --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 --out $O --tag _new > $O/new.log 2>&1 &
wait
$PY - << 'EOF'
import json
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/w1_miss"
for who in ("current", "new"):
    rows = [c for c in json.load(open(f"{O}/eval_teacher_{who}.json"))["clips"] if c["task"] == "ThreeStepPoleRun"]
    miss = np.zeros(3); n = 0; offs = {0: [], 1: [], 2: []}; fell_after_miss = 0; nm = 0
    for r in rows:
        d = r["detail"]
        for i, st in enumerate(d["stepped"]):
            n += 1
            for s in range(3):
                if not st[s]:
                    miss[s] += 1
                    offs[s].append(d["near_xy"][i][s])
            if not all(st[:3]):
                nm += 1
    print(f"== {who}: rollouts {n}; rollouts with a missed stake {100 * nm / n:.1f} %; miss rate per stake "
          + " / ".join(f"S{s + 1} {100 * miss[s] / n:.1f} %" for s in range(3)))
    for s in range(3):
        if offs[s]:
            a = np.array(offs[s])
            ok = a[np.abs(a).sum(1) < 5]
            if len(ok):
                print(f"   S{s + 1} missed ({len(a)}): closest landing along-course median {np.median(ok[:, 0]):+.2f} m "
                      f"[{np.percentile(ok[:, 0], 10):+.2f}, {np.percentile(ok[:, 0], 90):+.2f}], across median "
                      f"{np.median(np.abs(ok[:, 1])):.2f} m;  no loaded foot near: {len(a) - len(ok)}")
EOF
