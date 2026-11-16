#!/bin/bash
# Video: v6g10m seed-1 L0 vs L4 on the unseen test trial ThreeStepPoleRun_012, task criterion, nominal environment.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_DR=0 HNU_START_FRAC=0
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
S=$H/runs/students_v6g10m_d2_s1
CLIP=${1:-ThreeStepPoleRun_012}
CUDA_VISIBLE_DEVICES=$G $PY tools/record_task.py --clip $CLIP --tries 15 --out $H/runs/videos/task_${CLIP}_L0_vs_L4.mp4 \
  --student "L0=$S/Z0/student_2000.pt:$H/data/z_tables_v5t/Z0:success_miss" \
  --student "L4=$S/Z4/student_2000.pt:$H/data/z_tables_v5t/Z4:success_clean"
