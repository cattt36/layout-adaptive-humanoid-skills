#!/bin/bash
# Post-spacing sensitivity of a bridge teacher bank (default: the proposed v6g10n bank), corrected geometry, nominal,
# 16 rollouts per clip, all CrouchedTraverse trials: posts moved together towards (-) / away from (+) the beam centre by
# 5 / 10 cm, and independently along the beam by +-5 / +-10 cm.     usage: bridge_posts.sh [bank file] [out tag]
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
BANK=${1:-$H/runs/night/v6g10n_teachers_proposed.txt}
O=$H/runs/bridge_posts${2:+_$2}
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 \
  HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 HNU_DR=0
CK=$(cat $BANK)
mkdir -p $O
G=0
for C in "spread -0.10" "spread -0.05" "spread 0.05" "spread 0.10" "jitter 0.05" "jitter 0.10"; do
  set -- $C
  if [ $1 = spread ]; then V="HNU_BRIDGE_POST_SPREAD=$2"; else V="HNU_BRIDGE_POST_JITTER=$2"; fi
  env $V CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O --tag _$1$2 > /dev/null 2>&1 &
  G=$((1 - G)); [ $G = 0 ] && wait
done
wait
$PY - "$O" << 'EOF'
import json, sys
O = sys.argv[1]
print("post spacing -> success % (contact-free %), all CrouchedTraverse trials, nominal")
for tag, lab in (("spread-0.10", "posts 10 cm closer to the centre"), ("spread-0.05", "posts 5 cm closer"),
                 ("spread0.05", "posts 5 cm further out"), ("spread0.10", "posts 10 cm further out"),
                 ("jitter0.05", "each post +-5 cm along the beam"), ("jitter0.10", "each post +-10 cm")):
    s = json.load(open(f"{O}/eval_teacher_{tag}.json"))["summary"]["CrouchedTraverse/all"]
    print(f"   {lab:36s} {100 * s['success']:.0f} ({100 * s['success_no_contact']:.0f})")
EOF
