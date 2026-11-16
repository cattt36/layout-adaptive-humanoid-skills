#!/bin/bash
# Representative L0 / L4 videos (v6g10m seed 1) on all 17 test trials, first rollout each, task criterion, nominal.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_DR=0 HNU_START_FRAC=0 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
S=$H/runs/students_v6g10m_d2_s1
CUDA_VISIBLE_DEVICES=$G $PY tools/record_batch.py --split test ${1:+--seed $1} \
  --student "L0=$S/Z0/student_2000.pt:$H/data/z_tables_v5t/Z0" \
  --student "L4=$S/Z4/student_2000.pt:$H/data/z_tables_v5t/Z4"
