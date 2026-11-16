#!/bin/bash
# Videos of the stake teachers h10c (0-20 ms latency-robust) on 0.10 m stakes, robot-scale stride, nominal, from frame 0.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl PYTHONPATH=$H/code HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 \
  HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_DR=0
O=$H/videos/teachers_h10c; mkdir -p $O
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h10; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h10
  CK=$(cat $H/runs/stakes_eval/robust_stakes_h10c/$T/checkpoint.txt)
  for C in $(ls $D/$T/*_motion.npz | sed -n '2p;12p;22p' | xargs -n1 basename | sed 's/_motion.npz//'); do
    CUDA_VISIBLE_DEVICES=1 HNU_STAKES_ROOT=$D $PY tools/record_expert.py HNU-Track-$T-Stakes $C $CK \
      $O/${C}_stakes_0.10m.mp4 "$T teacher h10c on 0.10 m stakes" 2>&1 | tail -n 1
  done
done
echo done
