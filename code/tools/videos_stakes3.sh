#!/bin/bash
# stakes3 expert videos (stake_missed termination disabled so the whole attempt is visible).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export MUJOCO_GL=egl PYTHONPATH=$H/code
S=$H/runs/stakes_eval/stakes3
for T in ThreeStepPoleRun FiveStepPoleRun; do
  CK3=$(grep $T $S/checkpoints.txt | awk '{print $2}')
  C=$(ls $H/data/ref_stakes/$T/*_motion.npz | sed -n 3p | xargs basename | sed 's/_motion.npz//')
  CUDA_VISIBLE_DEVICES=1 HNU_STAKE_MISS_TERM=0 $PY tools/record_stakes.py $T $C $CK3 $H/videos/stakes3/${T}_${C}.mp4 >> $H/videos/stakes3/log.txt 2>&1
done
$PY tools/stakes_summary.py "$S/W1_clean/*.json" >> $H/videos/stakes3/log.txt 2>&1
echo done >> $H/videos/stakes3/log.txt
