#!/bin/bash
# Stakes round 3: the relaxed-first-stake training (training only; evaluation strict) helped W2 (2.2% -> 11.6%), so
#   W1: perc_h10 + first stake relaxed, 5000 its   (GPU1)
#   W2: continue perc2_h10_skip1 for 5000 more its  (GPU1, after W1 to keep the GPU shared with the student seeds)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
run() {  # T gpu src run data_dir
  local T=$1 G=$2 SRC=$3 RUN=$4 D=$5
  ( cd $H/runs && CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=1 HNU_STAKES_ROOT=$D $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 5000 \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/$RUN/$T
  mkdir -p $O
  for X in "" "--push --noise"; do
    CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 \
      --motion-dir $D/$T --stake-top 0.10 $X --out $O >> $O/eval.log 2>&1
  done
}
run ThreeStepPoleRun 1 $(last ThreeStepPoleRun perc_h10) perc3_h10_skip1 $H/data/ref_stakes_s93_h10
run FiveStepPoleRun 1 $(last FiveStepPoleRun perc2_h10_skip1) perc3_h10_skip1 $H/data/ref_stakes_s83_h10
$PY tools/stakes_summary.py "$H/runs/stakes_eval/perc3_*/*/*.json" > $H/runs/stakes_eval/perc3_summary.txt 2>&1
