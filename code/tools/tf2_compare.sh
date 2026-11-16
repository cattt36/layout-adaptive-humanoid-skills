#!/bin/bash
# Episode statistics: end of the current teachers' training (rs10c) vs start / end of the v2 pilot.
L=/root/gpufree-data/hnu_v2/runs/logs
for T in ThreeStepPoleRun FiveStepPoleRun; do
  for R in rs10c tf2_pilot; do
    f=$L/${T}_$R.log
    [ -f $f ] || { echo "$T $R: no log"; continue; }
    echo "== $T $R first / last"
    grep -E 'Mean episode length' $f | head -1
    grep -E 'Episode_Termination/stake_missed:' $f | head -1
    grep -E 'Mean episode length' $f | tail -1
    grep -E 'Episode_Termination/(stake_missed|ee_body_pos|anchor_xy|anchor_pos):' $f | tail -4
  done
done
