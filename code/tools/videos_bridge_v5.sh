#!/bin/bash
# Grid videos (teacher + L0-L4) of the current bridge-scene students: v5 + DAgger round 2, RL seed 1.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
O=$H/videos/bridge_v5_d2
mkdir -p $O
for C in CrouchedTraverse_007 CrouchedTraverse_012 CrouchedTraverse_021 CrouchedTraverse_028; do
  CUDA_VISIBLE_DEVICES=1 $PY record_grid.py --clip $C --students-root $H/runs/students_v5_d2 --ckpt-name student_2000.pt \
    --title "(v5+DAgger2)" --z-root $H/data/z_tables_v5 $CK --out $O/grid_$C.mp4 2>&1 | grep -a -e "->" -e Error
done
