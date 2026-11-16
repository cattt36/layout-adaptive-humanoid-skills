#!/bin/bash
# Fine-tune the flat CrouchedTraverse expert under the marker-defined bridge, then evaluate (clean, push+noise).
# usage: train_bridge.sh <gpu> [iters] [run_name]
G=${1:-0}; ITERS=${2:-6000}; RUN=${3:-bridge1}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
T=CrouchedTraverse
SRC=$(cat $H/runs/expert_eval/$T/checkpoint.txt)
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
echo done > $O/DONE
