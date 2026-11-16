#!/bin/bash
# Restart the bridge-scene students with <= 2 jobs per GPU.
H=/root/gpufree-data/hnu_v2
for p in $(ps -eo pid,args | grep 'run_bridge_students_[r].sh' | awk '{print $1}'); do kill $p; done
pkill -f 'distill_train.py --profile v6r2'
sleep 5
cd $H/code
setsid nohup bash run_bridge_students_r.sh > /dev/null 2>&1 < /dev/null &
sleep 240
ps -eo args | grep 'distill_trai[n]' | sed -E 's/.*--profile ([a-z0-9_]+).*/\1/'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -n 2 $H/runs/night/v6r2.log
