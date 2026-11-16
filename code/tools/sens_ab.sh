#!/bin/bash
# A/B: do the extra foot-stake contact sensors change the rollouts?  Same clips, same seeds, L4 v6g10m s1, 3 seeds.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_DR=0 HNU_START_FRAC=0 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json
S=$H/runs/students_v6g10m_d2_s1
for SEED in 1000 1001 1002; do
  for MODE in with without; do
    if [ $MODE = without ]; then export HNU_REC_NO_STAKE_SENS=1; else unset HNU_REC_NO_STAKE_SENS; fi
    echo "== seed $SEED sensors $MODE"
    CUDA_VISIBLE_DEVICES=0 $PY tools/record_batch.py --clips FiveStepPoleRun_013,ThreeStepPoleRun_012 --seed $SEED --slow 50 \
      --out $H/runs/videos/ab_$MODE --student L4=$S/Z4/student_2000.pt:$H/data/z_tables_v5t/Z4 2>&1 | grep -E 'SUCCESS|FELL|did not|Error'
  done
done
