#!/bin/bash
# Sanity check of the bridge offset plumbing: a large beam height error (+-15 cm) and shift (+-20 cm) must change the
# outcome (contacts / failures) compared with no offset.  New bridge teacher, nominal, 8 rollouts per clip.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0
NEWDU=$(ls $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_tf2_full/model_*.pt | sort -V | tail -1)
CUR=$(cat $H/runs/night/v6g10t_teachers.txt)
OLDDU=$(echo $CUR | sed 's/ /\n/g' | grep -A1 -- "--ckpt-CrouchedTraverse" | tail -1)
NEW=${CUR/$OLDDU/$NEWDU}
HNU_BRIDGE_JITTER_Z=0.15 CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $NEW --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 8 --out $H/runs/bridge_tol/z0.15 --tag _new > /dev/null 2>&1 &
HNU_BRIDGE_JITTER_XY=0.20 CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $NEW --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 8 --out $H/runs/bridge_tol/xy0.20 --tag _new > /dev/null 2>&1 &
wait
$PY - << 'EOF'
import json
H = "/root/gpufree-data/hnu_v2/runs/bridge_tol"
for d in ("xy0_z0", "z0.15", "xy0.20"):
    s = json.load(open(f"{H}/{d}/eval_teacher_new.json"))["summary"]["CrouchedTraverse/all"]
    print(f"{d:8s}: success {100 * s['success']:.0f} %  no-contact success {100 * s['success_no_contact']:.0f} %")
EOF
