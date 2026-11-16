#!/bin/bash
# Relaunch only the bridge-perception expert (perception.py broadcast fix); the padded checkpoint already exists.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
T=CrouchedTraverse
RUN=bridge_video_lift_perc
export HNU_PERCEPTION=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_BRIDGE_W=-5.0
rm -rf $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN
PD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_pad_bridge_video_lift | tail -1)
START=$(ls $PD/model_*.pt | tail -1)
( cd $H/runs && CUDA_VISIBLE_DEVICES=1 $PY $H/code/train_hnu.py HNU-Track-$T-Bridge --env.scene.num-envs 4096 \
    --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 5000 \
    --agent.resume True --agent.load-run $(basename $PD) --agent.load-checkpoint $(basename $START) \
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
sleep 150
echo "-- $(grep -a 'Learning iteration' $H/runs/logs/${T}_$RUN.log | tail -n 1 | tr -s ' ')"
grep -a -e Error -e perception $H/runs/logs/${T}_$RUN.log | head -n 3
