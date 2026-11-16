#!/bin/bash
# usage: launch_expert.sh <Task> <gpu> [num_envs] [iters] [run_name] [extra args...]
T=$1; G=$2; N=${3:-4096}; I=${4:-10000}; R=${5:-v1}; shift 5
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
mkdir -p $H/runs/logs
cd $H/runs
CUDA_VISIBLE_DEVICES=$G nohup $PY $H/code/train_hnu.py HNU-Track-$T \
  --env.scene.num-envs $N --env.commands.motion.motion-file $H/data/ref_v1/$T \
  --agent.max-iterations $I --agent.run-name $R "$@" > $H/runs/logs/${T}_${R}.log 2>&1 &
echo "launched $T on gpu $G pid $!"
