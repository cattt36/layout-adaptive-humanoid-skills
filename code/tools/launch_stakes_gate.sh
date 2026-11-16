#!/bin/bash
# Stakes, round "gate" (0.06 m, original run-up, stake jumps unchanged).
# Diagnosis: every stake-trained expert fails ~0.3 s after the run-up starts, even on near-flat data (completes
# 8-10 % vs 99.5 % for the flat expert) -> forcing world-frame tracking during the run-up (which the robot cannot
# follow at the human's acceleration) damaged the run-up.  Now the run-up is tracked body-relative only; the
# world-frame rewards and the drift limit switch on 0.8 s before the first stake, where the perceived stake guides
# the approach (+ direct foot-on-stake reward).  Two starts per task: the flat expert (padded for perception) and
# the current 0.06 m expert.  6000 its; strict evaluation (all stakes, falls).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
padflat() {  # T -> padded flat expert path
  local SRC=$(cat $H/runs/expert_eval/$1/checkpoint.txt)
  local OD=$H/runs/logs/rsl_rl/hnu_track_$1/$(date +%Y-%m-%d_%H-%M-%S)_pad_flat
  mkdir -p $OD; $PY tools/pad_checkpoint.py $SRC $OD/$(basename $SRC) 10 10 > /dev/null; echo $OD/$(basename $SRC)
}
run() {  # T gpu src run data skip_first
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
D1=$H/data/ref_stakes_s93_h06; D2=$H/data/ref_stakes_s83_h06
F1=$(padflat ThreeStepPoleRun); F2=$(padflat FiveStepPoleRun)
run ThreeStepPoleRun 1 $F1 gate_flat $D1 0 &
run FiveStepPoleRun 1 $F2 gate_flat $D2 1 &
run ThreeStepPoleRun 1 $(last ThreeStepPoleRun h06main) gate_h06 $D1 0 &
run FiveStepPoleRun 1 $(last FiveStepPoleRun h06main) gate_h06 $D2 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/gate_*/*/*.json" > $H/runs/stakes_eval/gate_summary.txt 2>&1
