#!/bin/bash
# Progress: stakes round 2, v5 extra seeds, v5b.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== stakes round 2"; cat $H/runs/stakes_eval/perc2_summary.txt 2>/dev/null || bash tools/stakes2_status.sh
echo "== v5 seeds"; cat $H/runs/night/v5_seeds.log 2>/dev/null
for s in 2 3; do for p in Z0 Z1 Z2 Z3 Z4; do
  echo "s$s $p v5:$(tail -n 1 $H/runs/students_v5_s$s/$p/train_log.jsonl 2>/dev/null | cut -c1-12) d2:$(tail -n 1 $H/runs/students_v5_d2_s$s/$p/train_log.jsonl 2>/dev/null | cut -c1-12)"
done; done
echo "== v5b"; cat $H/runs/night/v5b.log 2>/dev/null
$PY tools/bridge_table.py $H/runs/results_v5_s2 $H/runs/results_v5_d2_s2 $H/runs/results_v5_s3 $H/runs/results_v5_d2_s3 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /root/gpufree-data | tail -n 1
