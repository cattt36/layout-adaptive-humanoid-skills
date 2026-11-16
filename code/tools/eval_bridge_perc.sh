#!/bin/bash
# Re-run the evaluation of the bridge-perception expert (first attempt produced no output).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
export HNU_PERCEPTION=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_BRIDGE_W=-5.0
O=$H/runs/bridge_eval/bridge_video_lift_perc
CK=$(cat $O/checkpoint.txt)
cd $H/code
CUDA_VISIBLE_DEVICES=0 $PY eval_expert.py --task CrouchedTraverse --bridge --ckpt $CK --per-clip 8 --out $O > $O/clean.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY eval_expert.py --task CrouchedTraverse --bridge --ckpt $CK --per-clip 8 --push --noise --out $O > $O/push.log 2>&1
grep -a -e Error -e Traceback $O/clean.log | head -n 5
timeout 60 $PY tools/stakes_summary.py "$O/*.json"
for C in CrouchedTraverse_007 CrouchedTraverse_021; do
  MUJOCO_GL=egl PYTHONPATH=$H/code CUDA_VISIBLE_DEVICES=0 $PY tools/record_expert.py HNU-Track-CrouchedTraverse-Bridge $C $CK \
    $H/videos/geometry/bridge_video_lift_perc_$C.mp4 "CrouchedTraverse expert with bridge perception" 2>&1 | tail -n 1
done
