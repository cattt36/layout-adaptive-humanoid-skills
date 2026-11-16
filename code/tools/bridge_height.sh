#!/bin/bash
# Beam height sensitivity of the (new) bridge teacher with the corrected geometry (12-gon post prisms):
# fixed beam height error -8 / -5 / -3 / 0 / +3 / +5 / +8 cm (posts unchanged), nominal and randomised,
# 16 rollouts per clip, all CrouchedTraverse trials; success and contact-free success.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 \
  HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
CK=$(cat $H/runs/night/v6g10n_teachers_proposed.txt)
O=$H/runs/bridge_height
mkdir -p $O
G=0
for DRV in 0 1; do
  for DZ in -0.08 -0.05 -0.03 0 0.03 0.05 0.08; do
    HNU_DR=$DRV HNU_BRIDGE_DZ=$DZ CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom \
      --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 \
      --out $O/dr$DRV --tag _dz$DZ > /dev/null 2>&1 &
    G=$((1 - G))
    [ $G = 0 ] && wait
  done
  wait
done
$PY - << 'EOF'
import json
O = "/root/gpufree-data/hnu_v2/runs/bridge_height"
print("bridge teacher, beam height error -> success % (contact-free success %), all CrouchedTraverse trials")
for dr in ("0", "1"):
    cells = []
    for dz in ("-0.08", "-0.05", "-0.03", "0", "0.03", "0.05", "0.08"):
        s = json.load(open(f"{O}/dr{dr}/eval_teacher_dz{dz}.json"))["summary"]["CrouchedTraverse/all"]
        cells.append(f"{float(dz) * 100:+.0f} cm: {100 * s['success']:.0f} ({100 * s['success_no_contact']:.0f})")
    print(("nominal    " if dr == "0" else "randomised ") + " | ".join(cells))
EOF
