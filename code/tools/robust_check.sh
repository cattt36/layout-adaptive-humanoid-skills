#!/bin/bash
# Robust bridge expert, evaluated in its own task (eval_expert), nominal (HNU_DR=0) and randomised (HNU_DR=1).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
for RUN in r2_mild_bridge r2_full_bridge; do
  CK=$(last CrouchedTraverse $RUN)
  for DRV in 0 1; do
    O=/tmp/robust_check/${RUN}_dr$DRV; mkdir -p $O
    HNU_DR=$DRV HNU_DR_LEVEL=2 CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task CrouchedTraverse --bridge --ckpt $CK --per-clip 4 --out $O > $O/log.txt 2>&1
    $PY -c "import json,glob; j=json.load(open(glob.glob('$O/*.json')[0])); print('$RUN dr$DRV', round(j['success_rate'],3), j['termination_causes'])"
  done
done
