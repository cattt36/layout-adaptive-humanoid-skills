#!/bin/bash
# Retention (original motion, nominal, from frame 0, strict tracking criterion of eval_expert.py) of the CURRENT
# teachers and the NEW (tf2_full) teachers vs the original experts, same settings for both (bridge geometry
# ref_bridge_video_lift -- the default in hnu_tasks is an older bridge).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
cur() { sed 's/ /\n/g' $H/runs/night/v6g10t_teachers.txt | grep -A1 -- "--ckpt-$1" | tail -1; }
new() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_tf2_$2/model_*.pt | sort -V | tail -1; }
OD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
OW2=$(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt)
echo "current DuMu:"; CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 --orig $OD --tuned $(cur CrouchedTraverse) &
sleep 2
echo "new DuMu:"; CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 --orig $OD --tuned $(new CrouchedTraverse full) &
wait
echo "current W2:"; CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task FiveStepPoleRun --mode stakes_flat --perc-tuned 1 --orig $OW2 --tuned $(cur FiveStepPoleRun) &
sleep 2
echo "new W2:"; CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task FiveStepPoleRun --mode stakes_flat --perc-tuned 1 --orig $OW2 --tuned $(new FiveStepPoleRun full) &
wait
