#!/bin/bash
# Side-by-side teacher | v3-L2 student videos on unseen-subject test clips (one per action) + one training clip each.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl
O=$H/runs/videos
mkdir -p $O
for C in ${CLIPS:-ThreeStepPoleRun_021 FiveStepPoleRun_022 CrouchedTraverse_004 ThreeStepPoleRun_001 FiveStepPoleRun_001 CrouchedTraverse_001}; do
  $PY record_video.py --clip $C --student $H/runs/students_v3/Z2/student_2500.pt --student-label "Student L2 (deployable obs + z_skill)" \
    --latent-dir $H/data/z_tables_v3/Z2 $CK --out $O/${C}_teacher_vs_studentL2.mp4 2>&1 | grep -E '^(Wu|Du)|Error|error' | tail -2
done
ls -la $O
