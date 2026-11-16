#!/bin/bash
# Stop the v6g10t student batch (superseded) and start the final v6g10j batch + its TensorBoard monitor.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "run_v6g10t.sh"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10t"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10t|results_v6g10t|profile v6g10t"); do kill $p; done
sleep 10
echo "left over v6g10t processes:"; pgrep -af "v6g10t" | grep -v launch_j || echo none
cd $H/code
setsid nohup bash run_v6g10j.sh > $N/run_v6g10j.out 2>&1 < /dev/null &
sleep 5
bash tools/restart_monitor.sh v6g10j
sleep 60
echo "--- batch log"; cat $N/v6g10j.log
echo "--- gpus"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "--- tensorboard"; pgrep -af "tensorboard" | head -3
