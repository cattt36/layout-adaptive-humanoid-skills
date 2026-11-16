#!/bin/bash
# Stakes curriculum v3: robot-scale stake spacing.  Stride factor = the flat expert's measured achieved/reference
# step length on flat ground (tools/stride_diag.sh): W1 0.93, W2 0.83.  Same 4 stages as curriculum v1, from the
# flat expert, no extra stake reward.
H=/root/gpufree-data/hnu_v2
cd $H/code
HSCALE=0.93 setsid nohup bash tools/stakes_curriculum.sh ThreeStepPoleRun 0 3000 curr3 "" > /dev/null 2>&1 < /dev/null &
HSCALE=0.83 setsid nohup bash tools/stakes_curriculum.sh FiveStepPoleRun 1 3000 curr3 "" > /dev/null 2>&1 < /dev/null &
sleep 200
grep -h -v -E 'warn|Warn' $H/runs/logs/stakes_curr3_*.log | tail -n 30
