#!/bin/bash
# Progress of the running stakes curriculum v3 and the USD bridge runs.
H=/root/gpufree-data/hnu_v2
date '+%H:%M'
grep -h '^\[' $H/runs/logs/stakes_curr3_*.log 2>/dev/null | cut -c1-120
for f in $(ls -t $H/runs/logs/*StepPoleRun_curr3_*.log $H/runs/logs/CrouchedTraverse_bridge_usd_*.log 2>/dev/null); do
  echo "-- $(basename $f) $(grep -a 'Learning iteration' $f | tail -n 1 | tr -s ' ')"
  grep -a -e 'Mean episode length' -e 'stake_missed' -e 'bridge_contact' $f | tail -n 3 | tr -s ' '
done
