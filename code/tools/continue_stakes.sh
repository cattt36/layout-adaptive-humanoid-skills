#!/bin/bash
# Continue stakes training from the latest checkpoint of run <src_run_name> into a new run <dst_run_name>.
# usage: continue_stakes.sh <Task> <gpu> <src_run_name> <dst_run_name> <iterations>
T=$1; G=$2; SRC=$3; DST=$4; I=$5
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
D=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$SRC | tail -1)
CK=$(ls $D/model_*.pt | sort -V | tail -1)
cd $H/runs
CUDA_VISIBLE_DEVICES=$G nohup $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
  --agent.experiment-name hnu_track_$T --agent.run-name $DST --agent.max-iterations $I \
  --agent.resume True --agent.load-run $(basename $D) --agent.load-checkpoint $(basename $CK) \
  > $H/runs/logs/${T}_${DST}.log 2>&1 < /dev/null &
echo "continue $T from $(basename $D)/$(basename $CK) -> $DST ($I its, gpu $G)"
