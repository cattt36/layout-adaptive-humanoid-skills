#!/bin/bash
# Landing-error diagnosis on 0.03 m stakes (curriculum stage p1 checkpoints) for W1 and W2.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
for T in ThreeStepPoleRun FiveStepPoleRun; do
  RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_curr_p1 | tail -1)
  CK=$(ls $RD/model_*.pt | sort -V | tail -1)
  O=$H/runs/stakes_eval/landing_diag/$T
  mkdir -p $O
  HNU_STAKE_H=0.03 HNU_STAKE_TOL=0.30 HNU_ANCHOR_XY=0.60 HNU_STAKES_ROOT=$H/data/ref_stakes_h03 CUDA_VISIBLE_DEVICES=1 \
    $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --motion-dir $H/data/ref_stakes_h03/$T --stake-top 0.03 \
    --save-rollouts --out $O > $O/eval.log 2>&1
  echo "== $T"
  $PY tools/stake_landing_error.py $O $H/data/ref_stakes_h03/$T
done
