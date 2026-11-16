#!/bin/bash
# Progress of the stakes curriculum and the bridge expert.
H=/root/gpufree-data/hnu_v2
date +%H:%M:%S
grep -h '^\[' $H/runs/logs/stakes_curr_ThreeStepPoleRun.log $H/runs/logs/stakes_curr_FiveStepPoleRun.log 2>/dev/null
for f in $(ls -t $H/runs/logs/ThreeStepPoleRun_curr_*.log $H/runs/logs/FiveStepPoleRun_curr_*.log $H/runs/logs/CrouchedTraverse_bridge*.log 2>/dev/null); do
  echo "-- $(basename $f): $(grep -a 'Learning iteration' $f | tail -n 1 | tr -s ' ' | cut -c1-60)"
  grep -a -E 'Mean episode length|Episode_Termination/(stake_missed|anchor_xy|bad_orientation|ee_body_pos)|Episode_Reward/(feet_global_pos|bridge_contact)' $f | tail -n 6 | tr -s ' '
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
