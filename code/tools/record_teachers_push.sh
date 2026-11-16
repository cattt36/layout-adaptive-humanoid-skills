#!/bin/bash
# Same as record_teachers.sh, with the evaluation pushes (where the old and new teachers differ most); 2 reset seeds.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_DR=0 HNU_START_FRAC=0 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_REC_PUSH=1
NEW=$H/runs/tf2_video_teachers.txt
CLIPS=ThreeStepPoleRun_011,ThreeStepPoleRun_029,CrouchedTraverse_014,CrouchedTraverse_016,CrouchedTraverse_024
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
for SEED in 2000 2001; do
  CUDA_VISIBLE_DEVICES=$G $PY tools/record_batch.py --clips $CLIPS --seed $SEED --out $H/runs/videos/teachers_v2_push_s$SEED \
    --student "OLD-teacher=TEACHER@$H/runs/night/v6g10t_teachers.txt:$H/data/z_tables_v5t/Z4" \
    --student "NEW-teacher=TEACHER@$NEW:$H/data/z_tables_v5t/Z4" 2>&1 | grep -E "SUCCESS|FELL|did not|Error|Traceback" | sed "s/^/seed $SEED  /"
done
