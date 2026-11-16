#!/bin/bash
# Main stake scene = 0.06 m (user decision ).  Perception experts, robot-scale spacing, 5000 its each:
#   W1: continue the perception expert at 0.06 m (perc_h06), strict first stake
#   W2: best 0.10 m expert (perc3_h10_skip1) trained at 0.06 m with the first stake relaxed (training only)
# Evaluation strict at 0.06 m (clean, push+noise).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
run() {  # T gpu src run data_dir skip_first
  local T=$1 G=$2 SRC=$3 RUN=$4 D=$5 SK=$6
  ( cd $H/runs && CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$D $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 5000 \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/$RUN/$T
  mkdir -p $O
  echo $CK > $O/checkpoint.txt
  for X in "" "--push --noise"; do
    CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 \
      --motion-dir $D/$T --stake-top 0.06 $X --out $O >> $O/eval.log 2>&1
  done
}
run ThreeStepPoleRun 1 $(last ThreeStepPoleRun perc_h06) h06main $H/data/ref_stakes_s93_h06 0 &
run FiveStepPoleRun 1 $(last FiveStepPoleRun perc3_h10_skip1) h06main $H/data/ref_stakes_s83_h06 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/h06main/*/*.json" > $H/runs/stakes_eval/h06main_summary.txt 2>&1
