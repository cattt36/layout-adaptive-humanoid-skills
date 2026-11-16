#!/bin/bash
# Robust stake teachers, third attempt.  Component test on the 0.10 m experts: PD gains / effort limits /
# link masses leave the success unchanged (88-95 %), latency is what breaks them (20 ms fixed: 36-42 %, 0-40 ms: 0 %).
# -> latency-focused robust training at the realistic G1 level: 0-20 ms in 0.5 s pieces (+ the mild dynamics
# randomisation), 8000 its from the 0.10 m experts, 25 % frame-0 resets.  Evaluation: zero latency, fixed 20 ms,
# 0-20 ms pieces, fixed 40 ms (stress test), each with mild dynamics; strict criterion; retention vs the flat expert.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_START_FRAC=0.25
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
both() {  # T data skip
  local T=$1 D=$2 SK=$3 SRC=$(cat $H/runs/stakes_eval/g2_h10/$1/checkpoint.txt) RUN=rs10c
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] || \
  ( cd $H/runs && HNU_DR=1 HNU_DR_LEVEL=1 HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=1 \
      $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN \
      --agent.max-iterations 8000 --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/robust_stakes_h10c/$T
  mkdir -p $O; echo $CK > $O/checkpoint.txt
  for C in "lat0 0 0" "lat20 1 1" "lat0-20 0 1" "lat40 2 2"; do
    set -- $C
    for X in "" "--push --noise"; do
      HNU_DR=1 HNU_DR_LEVEL=1 HNU_DR_LAG_MIN=$2 HNU_DR_LAG_MAX=$3 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D \
        CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --motion-dir $D/$T --stake-top 0.10 $X \
        --out $O/$1 >> $O/eval.log 2>&1
    done
  done
  CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task $T --mode stakes_flat --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) \
    --tuned $CK --perc-tuned 1 >> $H/runs/stakes_eval/robust_stakes_h10c/retention.txt
}
both ThreeStepPoleRun $H/data/ref_stakes_s93_h10 0 &
both FiveStepPoleRun $H/data/ref_stakes_s83_h10 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/robust_stakes_h10c/*/*/*.json" > $H/runs/stakes_eval/robust_stakes_h10c/summary.txt 2>&1
echo done >> $H/runs/stakes_eval/robust_stakes_h10c/summary.txt
