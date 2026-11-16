#!/bin/bash
# Latest teacher videos: ThreeStepPoleRun / FiveStepPoleRun stake experts (gate_h06, 0.06 m, perception) and the CrouchedTraverse bridge expert.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl PYTHONPATH=$H/code HNU_PERCEPTION=1
O=$H/videos/teachers_0404; mkdir -p $O
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h06; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h06
  CK=$(cat $H/runs/stakes_eval/gate_h06/$T/checkpoint.txt)
  for C in $(ls $D/$T/*_motion.npz | sed -n '2p;12p;22p' | xargs -n1 basename | sed 's/_motion.npz//'); do
    CUDA_VISIBLE_DEVICES=1 HNU_STAKE_H=0.06 HNU_STAKES_ROOT=$D $PY tools/record_expert.py HNU-Track-$T-Stakes $C $CK \
      $O/${C}_stakes_0.06m.mp4 "$T teacher on 0.06 m stakes (perception)" 2>&1 | tail -n 1
  done
done
CK=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
for C in CrouchedTraverse_007 CrouchedTraverse_012 CrouchedTraverse_021; do
  CUDA_VISIBLE_DEVICES=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift $PY tools/record_expert.py HNU-Track-CrouchedTraverse-Bridge $C $CK \
    $O/${C}_bridge.mp4 "CrouchedTraverse teacher under the bridge (perception)" 2>&1 | tail -n 1
done
