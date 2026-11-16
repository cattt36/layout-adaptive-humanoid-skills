#!/bin/bash
# 0.10 m stakes (the real-robot stake height) with the training design that solved 0.06 m: body-relative run-up,
# world-frame objectives 0.8 s before the first stake, perception, 25 % frame-0 resets.  From the 0.06 m experts
# (gate_h06): 0.08 m 3000 its -> 0.10 m 5000 its.  Strict evaluation at 0.10 m + retention check.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_START_FRAC=0.25
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
mk() {  # T scale height out
  [ -d $4/$1 ] || $PY stage1_reference/make_stakes.py --task $1 --ref $H/data/ref_v1/$1 --csv $H/data/scene/$1.csv --height $3 \
    --hscale $2 --out $4 > /dev/null 2>&1
  rm -f $4/$1/ThreeStepPoleRun_007_*
}
stage() {  # T gpu src run iters height data skip
  local T=$1 G=$2 SRC=$3 RUN=$4 IT=$5 HS=$6 D=$7 SK=$8
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && return
  ( cd $H/runs && HNU_STAKE_H=$HS HNU_STAKE_SKIP_FIRST=$SK HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py \
      HNU-Track-$T-Stakes --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
}
both() {  # T scale tag skip
  local T=$1 SC=$2 TG=$3 SK=$4
  mk $T $SC 0.08 $H/data/ref_stakes_${TG}_h08
  stage $T 1 $(cat $H/runs/stakes_eval/gate_h06/$T/checkpoint.txt) g2_h08 3000 0.08 $H/data/ref_stakes_${TG}_h08 $SK
  stage $T 1 $(last $T g2_h08) g2_h10 5000 0.10 $H/data/ref_stakes_${TG}_h10 $SK
  local CK=$(last $T g2_h10) O=$H/runs/stakes_eval/g2_h10/$T D=$H/data/ref_stakes_${TG}_h10
  mkdir -p $O; echo $CK > $O/checkpoint.txt
  for X in "" "--push --noise"; do
    HNU_STAKE_H=0.10 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task $T \
      --stakes --ckpt $CK --per-clip 8 --motion-dir $D/$T --stake-top 0.10 $X --out $O >> $O/eval.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task $T --mode stakes_flat --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) \
    --tuned $CK --perc-tuned 1 >> $H/runs/stakes_eval/g2_h10/retention.txt
}
both ThreeStepPoleRun 0.93 s93 0 &
both FiveStepPoleRun 0.83 s83 1 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/g2_h10/*/*.json" > $H/runs/stakes_eval/g2_h10/summary.txt 2>&1
echo done >> $H/runs/stakes_eval/g2_h10/summary.txt
