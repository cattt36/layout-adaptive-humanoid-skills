#!/bin/bash
# Teacher bank at 0.95 / 1.05x (mild tempo change), nominal + randomised, task criterion, 16 rollouts per clip.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_START_FRAC=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
for T in 095 105; do for DRV in 0 1; do
  R=$H/data/ts/ts$T O=$H/runs/ts_check/ts$T
  HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift HNU_DR=$DRV \
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $R/z_tables_v5t/Z4 --per-clip 16 --out $O/dr$DRV --tag _teacher > $O.teacher$DRV.log 2>&1 || tail -3 $O.teacher$DRV.log
done; done
$PY - << 'EOF'
import json, os
H = "/root/gpufree-data/hnu_v2/runs"
for dr in ("0", "1"):
    row = []
    for sp in ("090", "095", "100", "105", "110"):
        f = f"{H}/results_v6g10k_bank_teacher_dr{dr}/eval_teacher_teacher.json" if sp == "100" else f"{H}/ts_check/ts{sp}/dr{dr}/eval_teacher_teacher.json"
        s = json.load(open(f))["summary"]
        row.append(f"x{int(sp) / 100:.2f} " + "/".join(f"{100 * s[f'{t}/train']['success']:.0f}" for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")))
    print(("nominal    " if dr == "0" else "randomised ") + " | ".join(row))
EOF
