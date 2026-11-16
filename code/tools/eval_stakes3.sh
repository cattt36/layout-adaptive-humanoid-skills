#!/bin/bash
# Wait until the stakes3 runs stop writing checkpoints, then evaluate the final checkpoint (clean and push+noise).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
RUN=${1:-stakes3}
cd $H/code
for T in ThreeStepPoleRun FiveStepPoleRun; do
  L=$H/runs/logs/${T}_$RUN.log
  for i in $(seq 1 240); do
    grep -aq 'Learning iteration 17997/17998' $L 2>/dev/null && break
    sleep 60
  done
done
sleep 120
O=$H/runs/stakes_eval/$RUN
mkdir -p $O
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$(ls -d $H/runs/logs/rsl_rl/hnu_track_${T}/*_$RUN | tail -1)
  CK=$(ls $D/model_*.pt | sort -V | tail -1)
  echo "$T $CK" >> $O/checkpoints.txt
  g=0; [ $T = FiveStepPoleRun ] && g=1
  ( CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --out $O/${T}_clean.json > $O/${T}_clean.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 8 --push --noise --out $O/${T}_push.json > $O/${T}_push.log 2>&1 ) &
done
wait
echo done > $O/DONE
