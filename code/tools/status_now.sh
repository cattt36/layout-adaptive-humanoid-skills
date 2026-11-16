#!/bin/bash
# Progress of the run-up stake training and v6r.
H=/root/gpufree-data/hnu_v2
date '+%m-%d %H:%M'
for f in ThreeStepPoleRun_runup_h06 FiveStepPoleRun_runup_h06 CrouchedTraverse_robust_bridge ThreeStepPoleRun_robust_flat FiveStepPoleRun_robust_flat; do
  echo "$f: $(grep -a 'Learning iteration' $H/runs/logs/$f.log 2>/dev/null | tail -n 1 | tr -s ' ')"
  grep -a -e 'Mean episode length' -e 'stake_missed' -e 'anchor_xy' $H/runs/logs/$f.log 2>/dev/null | tail -n 3 | tr -s ' '
done
cat $H/runs/night/v6r.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
