#!/bin/bash
# Progress bridge students (corrected latency), robust 0.10 m stake teachers, queued v6g10.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== bridge students"; tail -n 2 $H/runs/night/v6r2.log
for d in students_v6r2_d3_s1 students_v6r2_s2 students_v6r2_d2_s2 students_v6r2_s3 students_v6r2_d2_s3; do
  for p in Z0 Z1 Z2 Z3 Z4; do
    a=$(tail -n 1 $H/runs/$d/$p/train_log.jsonl 2>/dev/null | grep -o '"it": [0-9]*' | cut -d' ' -f2)
    [ -n "$a" ] && echo "$d $p it $a"
  done
done
for s in 1 2; do for DRV in 0 1; do
  [ -d $H/runs/results_v6r2_s${s}_dr$DRV ] && { echo "seed $s dr$DRV"; $PY tools/bridge_err.py $H/runs/results_v6r2_s${s}_dr$DRV; }
done; done
echo "== robust 0.10 m stakes"
for f in ThreeStepPoleRun_rs10b_mild FiveStepPoleRun_rs10b_mild ThreeStepPoleRun_rs10b_full FiveStepPoleRun_rs10b_full; do
  L=$H/runs/logs/$f.log; [ -f $L ] || continue
  echo "$f: $(grep -a 'Learning iteration' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f4) | ep $(grep -a 'Mean episode length' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f5)"
done
cat $H/runs/stakes_eval/robust_stakes_h10b/summary.txt $H/runs/stakes_eval/robust_stakes_h10b/retention.txt 2>/dev/null
echo "== v6g10"; tail -n 1 $H/runs/night/v6g10.log
