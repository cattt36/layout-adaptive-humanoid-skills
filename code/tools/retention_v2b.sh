#!/bin/bash
# Sequential, labelled retention runs (current vs new) -- the parallel run of retention_v2.sh printed in completion order.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
cur() { sed 's/ /\n/g' $H/runs/night/v6g10t_teachers.txt | grep -A1 -- "--ckpt-$1" | tail -1; }
new() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_tf2_$2/model_*.pt | sort -V | tail -1; }
OD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
OW2=$(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt)
for who in current new; do
  if [ $who = current ]; then TD=$(cur CrouchedTraverse); TW=$(cur FiveStepPoleRun); else TD=$(new CrouchedTraverse full); TW=$(new FiveStepPoleRun full); fi
  echo "$who DuMu: $(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 --orig $OD --tuned $TD 2>/dev/null | tail -1)" &
  echo "$who W2:   $(CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task FiveStepPoleRun --mode stakes_flat --perc-tuned 1 --orig $OW2 --tuned $TW 2>/dev/null | tail -1)" &
  wait
done
