#!/bin/bash
# Bridge placement tolerance of the bridge teacher (new tf2_full vs current), nominal, criterion v2 (DuMu unchanged:
# no fall + completed; contact allowed), 16 rollouts per clip, all CrouchedTraverse trials.
# Offsets: uniform horizontal shift of the whole bridge (+-XY) and beam height error (+-Z, posts unchanged).
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
mkdir -p $H/runs/bridge_tol
G=1
for C in "0 0" "0.03 0" "0.05 0" "0 0.02" "0 0.04" "0.05 0.04"; do
  set -- $C
  for who in old new; do
    CK=$CUR; [ $who = new ] && CK=$NEW
    HNU_BRIDGE_JITTER_XY=$1 HNU_BRIDGE_JITTER_Z=$2 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom \
      --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 \
      --out $H/runs/bridge_tol/xy$1_z$2 --tag _$who > /dev/null 2>&1 &
    G=$((1 - G))
  done
  wait
done
$PY - << 'EOF'
import json
H = "/root/gpufree-data/hnu_v2/runs/bridge_tol"
print("bridge teacher, nominal, all CrouchedTraverse trials: success % (no-contact success %)")
for xy, z in (("0", "0"), ("0.03", "0"), ("0.05", "0"), ("0", "0.02"), ("0", "0.04"), ("0.05", "0.04")):
    row = []
    for who in ("old", "new"):
        s = json.load(open(f"{H}/xy{xy}_z{z}/eval_teacher_{who}.json"))["summary"]["CrouchedTraverse/all"]
        row.append(f"{who}: {100 * s['success']:.0f} ({100 * s['success_no_contact']:.0f})")
    print(f"shift +-{float(xy) * 100:.0f} cm, height +-{float(z) * 100:.0f} cm:  " + "   ".join(row))
EOF
