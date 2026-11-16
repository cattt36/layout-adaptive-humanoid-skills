#!/bin/bash
# Videos: teacher + L0-L4 students (v4d, RL seed 1) on one test clip per action; stakes3 experts (W1/W2)
# with the stake_missed termination disabled so the whole attempt is visible; also W1 stakes eval (OOM before).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
O=$H/videos/v4d
mkdir -p $O $H/videos/stakes3
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
for C in ThreeStepPoleRun_030 FiveStepPoleRun_027 CrouchedTraverse_007; do
  CUDA_VISIBLE_DEVICES=1 $PY record_grid.py --clip $C --students-root $H/runs/students_v4d --z-root $H/data/z_tables_v4d \
    $CK --out $O/grid_${C}.mp4 >> $O/log.txt 2>&1
done
S=$H/runs/stakes_eval/stakes3
for T in ThreeStepPoleRun FiveStepPoleRun; do
  CK3=$(grep $T $S/checkpoints.txt | awk '{print $2}')
  C=$(ls $H/data/ref_stakes/$T/*_motion.npz | sed -n 3p | xargs basename | sed 's/_motion.npz//')
  CUDA_VISIBLE_DEVICES=1 HNU_STAKE_MISS_TERM=0 $PY tools/record_stakes.py $T $C $CK3 $H/videos/stakes3/${T}_${C}.mp4 >> $O/log.txt 2>&1
done
CK3=$(grep ThreeStepPoleRun $S/checkpoints.txt | awk '{print $2}')
CUDA_VISIBLE_DEVICES=1 $PY eval_expert.py --task ThreeStepPoleRun --stakes --ckpt $CK3 --per-clip 8 --out $S/W1_clean > $S/ThreeStepPoleRun_clean.log 2>&1
echo done >> $O/log.txt
