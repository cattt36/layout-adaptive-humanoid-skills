#!/bin/bash
# User start the CrouchedTraverse students now; ThreeStepPoleRun / FiveStepPoleRun move to 0.10 m stakes.
#  - stop v6g (it would wait for 0.06 m robust stake teachers and kill v6r2 students) and the 0.06 m robust stake runs
#  - start the bridge-scene students (run_bridge_students_r.sh, GPU0)
#  - queue robust stake training at 0.10 m after the 0.10 m experts are evaluated (GPU1)
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(ps -eo pid,args | grep -e 'run_v6[g].sh' -e 'robust_stake[s].sh' -e 'run_after_v6[r].sh' | awk '{print $1}'); do kill $p; done
pkill -f 'run-name rs_'
sleep 3
echo "[$(date +%H:%M:%S)] stopped: stakes move to 0.10 m (v6g will be relaunched with 0.10 m teachers)" >> $H/runs/night/v6g.log
setsid nohup bash run_bridge_students_r.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash tools/robust_stakes_h10.sh > /dev/null 2>&1 < /dev/null &
V=v6r2 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 120
bash tools/running.sh 2>/dev/null | grep -v sed
