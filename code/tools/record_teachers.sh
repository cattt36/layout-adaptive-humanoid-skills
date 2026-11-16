#!/bin/bash
# Videos: current teacher bank (left) vs the new criterion-v2 teachers (right), same clips, same reset seed, nominal,
# half speed.  New bank: W2 / CrouchedTraverse from tf2_full (accepted), W1 from the latest tf2_w1b checkpoint (preview).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_DR=0 HNU_START_FRAC=0 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json
lastck() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_tf2_$2/model_*.pt | sort -V | tail -1; }
NEW=$H/runs/tf2_video_teachers.txt
echo "--ckpt-ThreeStepPoleRun $(lastck ThreeStepPoleRun w1b) --ckpt-FiveStepPoleRun $(lastck FiveStepPoleRun full) --ckpt-CrouchedTraverse $(lastck CrouchedTraverse full)" > $NEW
cat $NEW
CLIPS=ThreeStepPoleRun_002,ThreeStepPoleRun_011,ThreeStepPoleRun_029,FiveStepPoleRun_005,FiveStepPoleRun_019,CrouchedTraverse_014,CrouchedTraverse_016,CrouchedTraverse_024
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
CUDA_VISIBLE_DEVICES=$G $PY tools/record_batch.py --clips $CLIPS --out $H/runs/videos/teachers_v2 \
  --student "OLD-teacher=TEACHER@$H/runs/night/v6g10t_teachers.txt:$H/data/z_tables_v5t/Z4" \
  --student "NEW-teacher=TEACHER@$NEW:$H/data/z_tables_v5t/Z4" 2>&1 | grep -E "SUCCESS|FELL|did not|Error|Traceback|->"
