#!/bin/bash
# Experts with exteroceptive input ("perception" look-ahead of stakes / bridge), warm-started from the best
# proprioceptive experts by zero-padding.  Stakes: from curriculum-v3 stage p2 (robot-scale spacing), stages
# h06 -> h10.  Bridge: capture-video layout (beam raised), contact weight -5, from bridge_video_lift.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
pkill -f 'run-name bridge_video_lift_w2[0]'
pkill -f 'bridge_video_finetune.sh -20.[0]'
pad() {  # T run_suffix n_actor n_critic -> prints padded ckpt path
  local T=$1 SUF=$2 NA=$3 NC=$4
  local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$SUF | tail -1)
  local CK=$(ls $RD/model_*.pt | sort -V | tail -1)
  local OD=$H/runs/logs/rsl_rl/hnu_track_$T/$(date +%Y-%m-%d_%H-%M-%S)_pad_$SUF
  mkdir -p $OD
  $PY tools/pad_checkpoint.py $CK $OD/$(basename $CK) $NA $NC > /dev/null
  echo $OD/$(basename $CK)
}
export HNU_PERCEPTION=1
for T in ThreeStepPoleRun FiveStepPoleRun; do
  g=0; [ $T = FiveStepPoleRun ] && g=1
  SC=0.93; [ $T = FiveStepPoleRun ] && SC=0.83
  START=$(pad $T curr3_p2 10 10)
  HSCALE=$SC setsid nohup bash tools/stakes_curriculum.sh $T $g 4000 perc $START \
    "h06 0.06 0.17 0.40" "h10 0.10 0.17 0.40" > /dev/null 2>&1 < /dev/null &
done
# bridge
T=CrouchedTraverse
export HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_BRIDGE_W=-5.0
SRCD=$(dirname $(cat $H/runs/bridge_eval/bridge_video_lift/checkpoint.txt))
SUF=$(basename $SRCD | sed 's/^[0-9_-]*_//')
START=$(pad $T $SUF 7 7)
RUN=bridge_video_lift_perc
( cd $H/runs && CUDA_VISIBLE_DEVICES=1 $PY $H/code/train_hnu.py HNU-Track-$T-Bridge --env.scene.num-envs 4096 \
    --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 5000 \
    --agent.resume True --agent.load-run $(basename $(dirname $START)) --agent.load-checkpoint $(basename $START) \
    > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null
  RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN | tail -1); CK=$(ls $RD/model_*.pt | sort -V | tail -1)
  O=$H/runs/bridge_eval/$RUN; mkdir -p $O; echo $CK > $O/checkpoint.txt
  cd $H/code
  CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --out $O > $O/clean.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --push --noise --out $O > $O/push.log 2>&1
  for C in CrouchedTraverse_007 CrouchedTraverse_021; do
    MUJOCO_GL=egl PYTHONPATH=$H/code CUDA_VISIBLE_DEVICES=1 $PY tools/record_expert.py HNU-Track-$T-Bridge $C $CK \
      $H/videos/geometry/${RUN}_$C.mp4 "CrouchedTraverse expert with bridge perception" >> $H/videos/geometry/log.txt 2>&1
  done
  echo done > $O/DONE ) > /dev/null 2>&1 < /dev/null &
sleep 240
grep -h '^\[' $H/runs/logs/stakes_perc_*.log
for f in $H/runs/logs/ThreeStepPoleRun_perc_h06.log $H/runs/logs/FiveStepPoleRun_perc_h06.log $H/runs/logs/${T}_$RUN.log; do
  echo "-- $(basename $f): $(grep -a 'Learning iteration' $f | tail -n 1 | tr -s ' ')"; grep -a -e Error -e perception $f | head -n 3
done
