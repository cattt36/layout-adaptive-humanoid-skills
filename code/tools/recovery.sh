#!/bin/bash
# EXPLORATORY (not pre-registered): recovery after stumbles, v6g10m seed 1, L0-L4, nominal, 16 rollouts per clip.
# Stumble: ThreeStepPoleRun / FiveStepPoleRun = at least one stake not hit (reference foot on top < 50 % of its core window, only stakes whose
# window was reached); CrouchedTraverse = bridge contact.  Recovery = success given a stumble.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 HNU_EVAL_DETAIL=1
O=$H/runs/recovery
gpu() { G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0; }
for P in Z0 Z1 Z2 Z3 Z4; do
  [ -f $O/eval_student_$P.json ] && continue
  gpu; CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10m_d2_s1/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5t/$P --per-clip 16 --out $O --tag _$P > $O.$P.log 2>&1 || tail -5 $O.$P.log
done
$PY - << 'EOF'
import json
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/recovery"
print("test trials | per action: success %, stumble %, recovery % (success | stumble), clean success % (success | no stumble)")
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    rows = [r for r in json.load(open(f"{O}/eval_student_{P}.json"))["clips"] if r["split"] == "test"]
    cells = []
    for t in ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]:
        S, ST = [], []
        for r in rows:
            if r["task"] != t:
                continue
            d = r["detail"]
            for i, ok in enumerate(d["success"]):
                if t == "CrouchedTraverse":
                    st = d["touch_frame"][i] >= 0
                else:
                    win = d["stake_win"]
                    ff = d["fail_frame"][i]
                    reached = [k for k, w in enumerate(win) if w[1] >= 0 and (ff < 0 or ff > w[1])]
                    st = any(d["stake_on"][i][k] < 0.5 for k in reached)
                S.append(ok); ST.append(st)
        S, ST = np.array(S), np.array(ST)
        rec = 100 * S[ST].mean() if ST.any() else np.nan
        cln = 100 * S[~ST].mean() if (~ST).any() else np.nan
        cells.append(f"{t[:5]} succ {100 * S.mean():3.0f} stumble {100 * ST.mean():3.0f} recover {rec:3.0f} clean {cln:3.0f}")
    print(f"L{P[1]}: " + " | ".join(cells))
EOF
