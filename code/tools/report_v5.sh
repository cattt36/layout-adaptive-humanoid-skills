#!/bin/bash
# Snapshot of the bridge-scene student comparison (v4d, v5, v5 DAgger round 2).
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
cat $H/runs/night/v5.log; cat $H/runs/night/v5_d2.log 2>/dev/null
for d in students_bridge:2500 students_v5:4000 students_v5_d2:2000; do
  dir=${d%%:*}; it=${d##*:}
  for p in Z0 Z1 Z2 Z3 Z4; do echo "$dir $p $(ls $H/runs/$dir/$p 2>/dev/null | grep -c student_$it.pt) $(tail -n 1 $H/runs/$dir/$p/train_log.jsonl 2>/dev/null | cut -c1-40)"; done
done
$PY tools/bridge_table.py $H/runs/results_bridge $H/runs/results_v5 $H/runs/results_v5_d2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
