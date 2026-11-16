#!/bin/bash
# Relaunch the held-out pipeline (free-memory scheduler) and its TensorBoard monitor.
H=/root/gpufree-data/hnu_v2
cd $H/code
bash -n run_v6g10h.sh || exit 1
setsid nohup bash run_v6g10h.sh > /dev/null 2>&1 < /dev/null &
sleep 2
bash tools/restart_monitor.sh v6g10h > /dev/null 2>&1
sleep 60
tail -n 3 $H/runs/night/v6g10h.log
pgrep -af "run_v6g10h|distill_train.py --profile v6g10h|monitor_tb.py v6g10h" | cut -c1-110
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
