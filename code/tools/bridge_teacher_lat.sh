#!/bin/bash
# Robust bridge teacher under the deployment latency model (0-20 ms, 0.5 s pieces) and zero / fixed 20 ms latency.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0
CK=$(grep -o -e '--ckpt-CrouchedTraverse [^ ]*' $H/runs/night/v6r2_teachers.txt | cut -d' ' -f2)
for C in "lat0 0 0" "lat20 1 1" "lat0-20 0 1"; do
  set -- $C; O=/tmp/brlat/$1; mkdir -p $O
  HNU_DR=1 HNU_DR_LEVEL=1 HNU_DR_LAG_MIN=$2 HNU_DR_LAG_MAX=$3 CUDA_VISIBLE_DEVICES=0 $PY eval_expert.py --task CrouchedTraverse --bridge \
    --ckpt $CK --per-clip 8 --out $O > $O/log.txt 2>&1
  $PY -c "import json,glob; j=json.load(open(glob.glob('$O/*.json')[0])); print('$1 success', round(j['success_rate'],3))"
done
