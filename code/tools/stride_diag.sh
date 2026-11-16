#!/bin/bash
# Robot-achievable stride: flat experts on flat ground (no stakes), landing positions vs the reference stake plan,
# and the stakes-curriculum stage-p1 experts for comparison.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
for T in ThreeStepPoleRun FiveStepPoleRun; do
  g=0; [ $T = FiveStepPoleRun ] && g=1
  O=$H/runs/stakes_eval/stride_diag/$T
  mkdir -p $O
  CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --ckpt $(cat $H/runs/expert_eval/$T/checkpoint.txt) --per-clip 8 \
    --stake-json-dir $H/data/ref_stakes/$T --stake-top 0.0 --save-rollouts --out $O > $O/eval.log 2>&1 &
done
wait
for T in ThreeStepPoleRun FiveStepPoleRun; do
  echo "== $T flat expert, flat ground"
  HNU_LAND_TOP=0.0 $PY tools/stake_landing_error.py $H/runs/stakes_eval/stride_diag/$T $H/data/ref_stakes/$T 2>/dev/null
  echo "== $T curriculum p1 expert, 0.03 m stakes"
  $PY tools/stake_landing_error.py $H/runs/stakes_eval/landing_diag/$T $H/data/ref_stakes_h03/$T 2>/dev/null | head -n 1
done
