#!/bin/bash
# Relaunch the bridge students and the robust 0.10 m stake teachers with the corrected latency model.
H=/root/gpufree-data/hnu_v2
for p in $(ps -eo pid,args | grep -e 'run_bridge_students_[r].sh' -e 'robust_stakes_h1[0].sh' -e 'run_after_v6[r].sh' | awk '{print $1}'); do kill $p; done
pkill -f 'distill_train.py --profile v6r2'
pkill -f 'run-name rs10_'
sleep 5
echo "[$(date +%H:%M:%S)] relaunch: piecewise-constant latency model" >> $H/runs/night/v6r2.log
cd $H/code
setsid nohup bash run_bridge_students_r2.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash tools/robust_stakes_h10b.sh > /dev/null 2>&1 < /dev/null &
sleep 240
bash tools/gpu_procs.sh
