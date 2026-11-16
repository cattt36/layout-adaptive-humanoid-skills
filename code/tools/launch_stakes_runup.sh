#!/bin/bash
# Stakes, run-up modification (user the run-up may change, the stake jumps may not).
# Diagnosis: ThreeStepPoleRun / FiveStepPoleRun fail at 13-19 % of the clip, i.e. during the run-up (W1 accelerates to 2.8 m/s within 1.4 s;
# the robot lags > 0.4 m and the episode is terminated before the first stake).
# New references: run-up replayed with a rate rising from r0 = 0.45 to 1 (about 1.45x longer, gentler acceleration),
# the last 0.5 s of the approach and all stake jumps exactly as before; 0.06 m stakes, robot-scale spacing,
# perception.  Fine-tune the best 0.06 m experts for 6000 its; strict evaluation.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
mk() {  # T hscale out
  [ -d $3/$1 ] || $PY stage1_reference/make_stakes.py --task $1 --ref $H/data/ref_v1/$1 --csv $H/data/scene/$1.csv --height 0.06 \
    --hscale $2 --runup-r0 0.45 --out $3 > $H/runs/logs/make_stakes_runup_$1.log 2>&1
  rm -f $3/$1/ThreeStepPoleRun_007_*
}
run() {  # T gpu src run data_dir skip_first
  local T=$1 G=$2 SRC=$3 RUN=$4 D=$5 SK=$6
  ( cd $H/runs && CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$D $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 6000 \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/$RUN/$T
  mkdir -p $O; echo $CK > $O/checkpoint.txt
  for X in "" "--push --noise"; do
    CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 \
      --motion-dir $D/$T --stake-top 0.06 $X --out $O >> $O/eval.log 2>&1
  done
}
D1=$H/data/ref_stakes_s93_r45_h06; D2=$H/data/ref_stakes_s83_r45_h06
mk ThreeStepPoleRun 0.93 $D1; mk FiveStepPoleRun 0.83 $D2
run ThreeStepPoleRun 1 $(last ThreeStepPoleRun h06main) runup_h06 $D1 0 &
run FiveStepPoleRun 1 $(last FiveStepPoleRun h06main) runup_h06 $D2 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/runup_h06/*/*.json" > $H/runs/stakes_eval/runup_h06_summary.txt 2>&1
