#!/bin/bash
# After the bridge expert and the stakes curriculum finish: summary of the geometry evaluations + expert videos.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl PYTHONPATH=$H/code
V=$H/videos/geometry
mkdir -p $V
for i in $(seq 1 600); do [ -f $H/runs/bridge_eval/bridge1/DONE ] && break; sleep 60; done
CK=$(cat $H/runs/bridge_eval/bridge1/checkpoint.txt)
for C in CrouchedTraverse_007 CrouchedTraverse_021; do
  CUDA_VISIBLE_DEVICES=1 $PY tools/record_expert.py HNU-Track-CrouchedTraverse-Bridge $C $CK $V/bridge_expert_$C.mp4 \
    "CrouchedTraverse expert under the marker-defined bridge" >> $V/log.txt 2>&1
done
for T in ThreeStepPoleRun FiveStepPoleRun; do
  for i in $(seq 1 600); do grep -q 'curriculum finished' $H/runs/logs/stakes_curr_$T.log 2>/dev/null && break; sleep 60; done
done
for T in ThreeStepPoleRun FiveStepPoleRun; do
  RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_curr_h10 | tail -1)
  CK=$(ls $RD/model_*.pt | sort -V | tail -1)
  for C in $(ls $H/data/ref_stakes_h10/$T/*_motion.npz | sed -n '3p;20p' | xargs -n1 basename | sed 's/_motion.npz//'); do
    CUDA_VISIBLE_DEVICES=1 HNU_STAKE_H=0.10 HNU_STAKES_ROOT=$H/data/ref_stakes_h10 HNU_STAKE_MISS_TERM=0 \
      $PY tools/record_expert.py HNU-Track-$T-Stakes $C $CK $V/stakes_curriculum_$C.mp4 "$T expert on stakes (0.30 m x 0.10 m), curriculum" \
      >> $V/log.txt 2>&1
  done
done
$PY tools/stakes_summary.py "$H/runs/stakes_eval/curr_*/*/*.json" "$H/runs/bridge_eval/*/*.json" > $V/summary.txt 2>&1
echo done >> $V/log.txt
