#!/bin/bash
# Stakes teachers, round 2 (0.10 m, robot-scale spacing, perception), 5000 iterations each, then strict evaluation
# (evaluation never relaxes anything):
#   W1  a: continue the perception expert (perc_h10)                         GPU0
#   W2  b: continue perc_h10 with the first-stake miss termination relaxed   GPU1
#   W2  c: slowed reference (0.85x speed) from perc_h10, first stake relaxed GPU1
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
run() {  # T gpu src_ckpt run data_dir [extra env]
  local T=$1 G=$2 SRC=$3 RUN=$4 D=$5
  ( cd $H/runs && CUDA_VISIBLE_DEVICES=$G HNU_STAKES_ROOT=$D $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
      --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 5000 \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  local CK=$(last $T $RUN) O=$H/runs/stakes_eval/$RUN/$T
  mkdir -p $O
  for X in "" "--push --noise"; do
    CUDA_VISIBLE_DEVICES=$G HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 \
      --motion-dir $D/$T --stake-top 0.10 $X --out $O >> $O/eval.log 2>&1
  done
}
D1=$H/data/ref_stakes_s93_h10
D2=$H/data/ref_stakes_s83_h10
D3=$H/data/ref_stakes_s83_t85_h10
if [ ! -d $D3/FiveStepPoleRun ]; then
  $PY stage1_reference/make_stakes.py --task FiveStepPoleRun --ref $H/data/ref_v1/FiveStepPoleRun --csv $H/data/scene/FiveStepPoleRun.csv \
    --height 0.10 --hscale 0.83 --tscale 0.85 --out $D3 > $H/runs/logs/make_stakes_t85.log 2>&1
fi
W1=$(last ThreeStepPoleRun perc_h10); W2=$(last FiveStepPoleRun perc_h10)
run ThreeStepPoleRun 0 $W1 perc2_h10 $D1 &
HNU_STAKE_SKIP_FIRST=1 run FiveStepPoleRun 1 $W2 perc2_h10_skip1 $D2 &
HNU_STAKE_SKIP_FIRST=1 run FiveStepPoleRun 1 $W2 perc2_h10_slow $D3 &
wait
$PY tools/stakes_summary.py "$H/runs/stakes_eval/perc2_*/*/*.json" > $H/runs/stakes_eval/perc2_summary.txt 2>&1
