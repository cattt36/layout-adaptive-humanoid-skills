#!/bin/bash
# Progress of stakes round 2.
H=/root/gpufree-data/hnu_v2
for f in ThreeStepPoleRun_perc2_h10 FiveStepPoleRun_perc2_h10_skip1 FiveStepPoleRun_perc2_h10_slow; do
  echo "$f: $(grep -a 'Learning iteration' $H/runs/logs/$f.log 2>/dev/null | tail -n 1 | tr -s ' ')"
  grep -a -e 'Error' -e 'Traceback' $H/runs/logs/$f.log 2>/dev/null | head -n 2
  grep -a -e 'Mean episode length' -e 'stake_missed' $H/runs/logs/$f.log 2>/dev/null | tail -n 2 | tr -s ' '
done
cat $H/runs/stakes_eval/perc2_summary.txt 2>/dev/null
