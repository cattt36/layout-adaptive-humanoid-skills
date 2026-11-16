#!/bin/bash
# Stop curriculum v1 after the diagnosis (first-stake and short landings) and start v2 from the p1 checkpoints with
# the direct foot-to-stake reward (weight 5, std 0.08 m).
H=/root/gpufree-data/hnu_v2
pkill -f 'stakes_curriculu[m].sh'
pkill -f 'HNU-Track-(Three|Five)StepPoleRun-Stakes'
sleep 5
cd $H/code
for T in ThreeStepPoleRun FiveStepPoleRun; do
  g=0; [ $T = FiveStepPoleRun ] && g=1
  RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_curr_p1 | tail -1)
  CK=$(ls $RD/model_*.pt | sort -V | tail -1)
  HNU_STAKE_RW=5 setsid nohup bash tools/stakes_curriculum.sh $T $g 3000 curr2 $CK \
    "p2 0.03 0.17 0.40" "h06 0.06 0.17 0.40" "h10 0.10 0.17 0.40" > /dev/null 2>&1 < /dev/null &
done
sleep 120
grep -h '^\[' $H/runs/logs/stakes_curr2_*.log
for T in ThreeStepPoleRun FiveStepPoleRun; do grep -a -E 'Learning iteration|stake_foot' $H/runs/logs/${T}_curr2_p2.log | tail -n 2; done
