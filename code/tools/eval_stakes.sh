#!/bin/bash
# Evaluate the stakes experts (latest checkpoint of the stakes run) on the per-clip stakes scene, clean + push.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_stakes | tail -1)
  CK=$(ls $D/model_*.pt | sort -V | tail -1)
  O=$H/runs/expert_eval_stakes/$T
  mkdir -p $O
  echo $CK > $O/checkpoint.txt
  g=0; [ $T = FiveStepPoleRun ] && g=1
  ( CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --out $O > $O/eval_clean.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --push --noise --out $O > $O/eval_push.log 2>&1 ) &
done
wait
grep -h '^{' $H/runs/expert_eval_stakes/*/eval_*.log | cut -c1-40,170-330
