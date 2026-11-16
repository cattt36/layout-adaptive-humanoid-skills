#!/bin/bash
# One-shot progress report: stakes curriculum v2, bridge expert, label-scarce Q_exec, geometry videos.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== stakes curriculum v2"
grep -h '^\[' $H/runs/logs/stakes_curr2_*.log 2>/dev/null | cut -c1-140
for f in $(ls -t $H/runs/logs/*StepPoleRun_curr2_*.log 2>/dev/null | head -n 2); do
  echo "-- $(basename $f): $(grep -a 'Learning iteration' $f | tail -n 1 | tr -s ' ' | cut -c1-50)"
  grep -a -E 'Mean episode length|stake_missed|stake_foot' $f | tail -n 3 | tr -s ' '
done
$PY tools/stakes_summary.py "$H/runs/stakes_eval/curr2_*/*/*.json" 2>/dev/null
echo "== bridge"
ls $H/runs/bridge_eval/bridge1 2>/dev/null | tr '\n' ' '; echo
$PY tools/stakes_summary.py "$H/runs/bridge_eval/*/*.json" 2>/dev/null
grep -h -o '"bridge_contact_rate": [0-9.]*' $H/runs/bridge_eval/*/*.json 2>/dev/null
echo "== Q_exec label-scarce"
ls $H/runs/quality_aux_scarce_v4d 2>/dev/null | tr '\n' ' '; echo
cat $H/runs/quality_aux_scarce_v4d/frac*/table.md 2>/dev/null
echo "== videos"
ls $H/videos/geometry 2>/dev/null | tr '\n' ' '; echo
grep -a -e completed -e FELL $H/videos/geometry/log.txt 2>/dev/null | tail -n 4
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
