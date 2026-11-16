#!/bin/bash
# Fine-tune the flat ThreeStepPoleRun/FiveStepPoleRun experts on per-clip stakes (diameter 0.30 m, height 0.10 m, raised references).
# usage: launch_stakes.sh <Task> <gpu> [iters]
T=$1; G=$2; I=${3:-3000}; RN=${4:-stakes}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
CK=$(cat $H/runs/expert_eval/$T/checkpoint.txt)
RUN=$(basename $(dirname $CK))
cd $H/runs
CUDA_VISIBLE_DEVICES=$G nohup $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
  --agent.experiment-name hnu_track_$T --agent.run-name $RN --agent.max-iterations $I \
  --agent.resume True --agent.load-run $RUN --agent.load-checkpoint $(basename $CK) \
  > $H/runs/logs/${T}_${RN}.log 2>&1 < /dev/null &
echo "launched $T stakes from $RUN/$(basename $CK) on gpu $G"
