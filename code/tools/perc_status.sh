#!/bin/bash
# Status of the perception experts (stakes perc_*, bridge video_lift perc / w5).
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
date '+%H:%M'
grep -h '^\[' $H/runs/logs/stakes_perc_*.log 2>/dev/null | cut -c1-220
cd $H/code
timeout 120 $PY tools/stakes_summary.py "$H/runs/stakes_eval/perc_*/*/*.json" "$H/runs/bridge_eval/bridge_video_lift*/*.json"
for f in $(ls -t $H/runs/logs/*StepPoleRun_perc_*.log $H/runs/logs/CrouchedTraverse_bridge_video_lift_*.log 2>/dev/null); do
  echo "-- $(basename $f) $(grep -a 'Learning iteration' $f | tail -n 1 | tr -s ' ')"
  grep -a -e 'Mean episode length' -e 'stake_missed' -e 'bridge_contact' $f | tail -n 3 | tr -s ' '
done
