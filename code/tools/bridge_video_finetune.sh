#!/bin/bash
# video_lift bridge: the expert completes 88% but brushes the thin posts in ~90% of runs (contact penalty -1 too weak).
# Fine-tune from that expert with a stronger contact penalty, then evaluate and record.
W=${1:--5.0}; G=${2:-0}; ITERS=${3:-4000}; RUN=${4:-bridge_video_lift_w5}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
T=CrouchedTraverse
export HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_BRIDGE_W=$W
SRC=$(cat $H/runs/bridge_eval/bridge_video_lift/checkpoint.txt)
( cd $H/runs && CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Bridge --env.scene.num-envs 4096 \
    --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $ITERS \
    --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
    > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN | tail -1)
CK=$(ls $RD/model_*.pt | sort -V | tail -1)
O=$H/runs/bridge_eval/$RUN
mkdir -p $O
echo $CK > $O/checkpoint.txt
cd $H/code
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --out $O > $O/clean.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --push --noise --out $O > $O/push.log 2>&1
export MUJOCO_GL=egl PYTHONPATH=$H/code
for C in CrouchedTraverse_007 CrouchedTraverse_021; do
  CUDA_VISIBLE_DEVICES=$G $PY tools/record_expert.py HNU-Track-$T-Bridge $C $CK $H/videos/geometry/${RUN}_$C.mp4 \
    "CrouchedTraverse expert, capture-video posts, contact weight $W" >> $H/videos/geometry/log.txt 2>&1
done
echo done > $O/DONE
