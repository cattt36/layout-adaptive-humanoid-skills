#!/bin/bash
# Robust stake teachers (same procedure as the bridge / flat robust teachers):
#   gate_h06 experts (0.06 m, perception, body-relative run-up) -> mild DR 4000 its -> full DR 6000 its,
#   25 % resets at frame 0, same training objectives as gate_h06.
# Checks: strict stake evaluation nominal / randomised (clean, push+noise) + retention vs the flat expert.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_START_FRAC=0.25
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
stage() {  # T gpu src run iters level data skip_first
  local T=$1 G=$2 SRC=$3 RUN=$4 IT=$5 LV=$6 D=$7 SK=$8
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && return
  ( cd $H/runs && HNU_DR=1 HNU_DR_LEVEL=$LV HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=$G \
      $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN \
      --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
}
both() {  # T data skip
  local T=$1 D=$2 SK=$3
  stage $T 1 $(cat $H/runs/stakes_eval/gate_h06/$T/checkpoint.txt) rs_mild 4000 1 $D $SK
  stage $T 1 $(last $T rs_mild) rs_full 6000 2 $D $SK
  local CK=$(last $T rs_full) O=$H/runs/stakes_eval/robust_stakes/$T
  mkdir -p $O; echo $CK > $O/checkpoint.txt
  for DRV in 0 1; do for X in "" "--push --noise"; do
    HNU_DR=$DRV HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task $T --stakes \
      --ckpt $CK --per-clip 8 --motion-dir $D/$T --stake-top 0.06 $X --out $O/dr$DRV >> $O/eval.log 2>&1
  done; done
  CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task $T --mode stakes_flat --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) \
    --tuned $CK --perc-tuned 1 >> $H/runs/stakes_eval/robust_stakes/retention.txt
}
both ThreeStepPoleRun $H/data/ref_stakes_s93_h06 0 &
both FiveStepPoleRun $H/data/ref_stakes_s83_h06 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/robust_stakes/*/dr*/*.json" > $H/runs/stakes_eval/robust_stakes/summary.txt 2>&1
echo done >> $H/runs/stakes_eval/robust_stakes/summary.txt
