#!/bin/bash
# After the gated stake runs: retention of the original motion (near-flat stakes, from frame 0) vs the flat expert.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
for i in $(seq 1 600); do [ -f $H/runs/stakes_eval/gate_summary.txt ] && break; sleep 60; done
for T in ThreeStepPoleRun FiveStepPoleRun; do
  for RUN in gate_flat gate_h06; do
    CK=$(cat $H/runs/stakes_eval/$RUN/$T/checkpoint.txt)
    CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task $T --mode stakes_flat --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) \
      --tuned $CK --perc-tuned 1 | sed "s/^/$RUN /" >> $H/runs/stakes_eval/gate_retention.txt
  done
done
