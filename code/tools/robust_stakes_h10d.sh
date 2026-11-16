#!/bin/bash
# Latency-robust stake teachers with rehearsal of the original motion (h10d).  h10c reached 88.5 % (W1) / 91.5 % (W2)
# all-stakes success at 0-20 ms but failed the retention check (W1 100 -> 94 %, W2 88 -> 70 % on the original-stride
# near-flat motion).  Rehearsal: the training clip set = stake clips (0.10 m, robot-scale stride) + the original flat
# reference clips (no stakes, body-relative tracking like the flat expert), same latency model 0-20 ms, from h10c,
# 4000 its.  Evaluation as h10c + retention.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_START_FRAC=0.25
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
both() {  # T data skip gpu
  local T=$1 D=$2 SK=$3 G=$4 SRC=$(cat $H/runs/stakes_eval/robust_stakes_h10c/$1/checkpoint.txt) RUN=rs10d
  # rehearsal clip directory: stake clips + the original flat clips renamed <task>_9xx (no stakes json -> no stakes)
  local RH=${D}_rehearsal
  if [ ! -d $RH/$T ]; then
    mkdir -p $RH/$T
    for f in $D/$T/*; do ln -sf $f $RH/$T/; done
    for f in $H/data/ref_v1/$T/*_motion.npz; do n=$(basename $f _motion.npz); ln -sf $f $RH/$T/${T}_9${n##*_}_motion.npz; done
  fi
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] || \
  ( cd $H/runs && HNU_DR=1 HNU_DR_LEVEL=1 HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$RH CUDA_VISIBLE_DEVICES=$G \
      $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
      --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 4000 --agent.resume True \
      --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/robust_stakes_h10d/$T
  mkdir -p $O; echo $CK > $O/checkpoint.txt
  for C in "lat0 0 0" "lat20 1 1" "lat0-20 0 1"; do
    set -- $C
    for X in "" "--push --noise"; do
      HNU_DR=1 HNU_DR_LEVEL=1 HNU_DR_LAG_MIN=$2 HNU_DR_LAG_MAX=$3 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D \
        CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --motion-dir $D/$T --stake-top 0.10 $X \
        --out $O/$1 >> $O/eval.log 2>&1
    done
  done
  CUDA_VISIBLE_DEVICES=$G $PY tools/retention_check.py --task $T --mode stakes_flat --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) \
    --tuned $CK --perc-tuned 1 >> $H/runs/stakes_eval/robust_stakes_h10d/retention.txt
}
both ThreeStepPoleRun $H/data/ref_stakes_s93_h10 0 0 &
both FiveStepPoleRun $H/data/ref_stakes_s83_h10 1 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/robust_stakes_h10d/*/*/*.json" > $H/runs/stakes_eval/robust_stakes_h10d/summary.txt 2>&1
echo done >> $H/runs/stakes_eval/robust_stakes_h10d/summary.txt
