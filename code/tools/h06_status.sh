#!/bin/bash
# Progress / results of the 0.06 m stake experts.
H=/root/gpufree-data/hnu_v2
for T in ThreeStepPoleRun FiveStepPoleRun; do
  echo "$T $(grep -a 'Learning iteration' $H/runs/logs/${T}_h06main.log | tail -n 1 | tr -s ' ')"
done
cat $H/runs/stakes_eval/h06main_summary.txt 2>/dev/null
