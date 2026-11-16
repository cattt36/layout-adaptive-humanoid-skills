#!/bin/bash
# Resume stakes fine-tuning from the last checkpoint of the crashed (NaN) stakes run.
# usage: resume_stakes.sh <Task> <gpu> <until_iteration>
T=$1; G=$2; UNTIL=${3:-13000}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
D=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_stakes | tail -1)
CK=$(ls $D/model_*.pt | sort -V | tail -1)
IT=$(basename $CK .pt | sed 's/model_//')
cd $H/runs
CUDA_VISIBLE_DEVICES=$G nohup $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
  --agent.experiment-name hnu_track_$T --agent.run-name stakes --agent.max-iterations $((UNTIL - IT)) \
  --agent.resume True --agent.load-run $(basename $D) --agent.load-checkpoint $(basename $CK) \
  > $H/runs/logs/${T}_stakes_r2.log 2>&1 < /dev/null &
echo "resumed $T from $(basename $D)/$(basename $CK) for $((UNTIL - IT)) iterations on gpu $G"
