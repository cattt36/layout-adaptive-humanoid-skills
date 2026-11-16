#!/bin/bash
# Stage 3 per task: evaluate the final expert (clean + pushes) and save rollouts for the annotations.
# usage: run_post_expert.sh <Task> <gpu> [run_name]
T=$1; G=$2; R=${3:-v1}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
RUN=$(ls -d $H/runs/logs/rsl_rl/hnu_track_${T}/*_${R} | tail -1)
CK=$(ls $RUN/model_*.pt | sort -V | tail -1)
OUT=$H/runs/expert_eval/$T
mkdir -p $OUT
echo "$CK" > $OUT/checkpoint.txt
cd $H/code
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --ckpt $CK --per-clip 8 --save-rollouts --out $OUT > $OUT/eval_clean.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --ckpt $CK --per-clip 8 --push --noise --save-rollouts --out $OUT > $OUT/eval_push.log 2>&1
grep -E '^\{' $OUT/eval_clean.log $OUT/eval_push.log
