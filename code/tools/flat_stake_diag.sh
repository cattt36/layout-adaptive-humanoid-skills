#!/bin/bash
# Diagnostic: how precisely does the FLAT expert place its feet on the (virtual) stake positions on flat ground?
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
O=$H/runs/stakes_eval/flat_diag
mkdir -p $O
for T in ThreeStepPoleRun FiveStepPoleRun; do
  g=0; [ $T = FiveStepPoleRun ] && g=1
  CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --ckpt $(cat $H/runs/expert_eval/$T/checkpoint.txt) --per-clip 8 \
    --stake-json-dir $H/data/ref_stakes/$T --stake-top 0.0 --out $O/$T > $O/$T.log 2>&1 &
done
wait
$PY tools/stakes_summary.py "$O/*/*.json"
