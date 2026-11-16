#!/bin/bash
# Restart the TensorBoard monitor of a batch (fresh run dir).
V=${1:-v6g10s}
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "tools/monitor_tb.py $V"); do kill $p; done
sleep 1
rm -rf $H/runs/logs/rsl_rl/monitor_$V $H/runs/logs/rsl_rl/monitor_${V}_detail
cd $H/code
CUDA_VISIBLE_DEVICES= setsid nohup /root/gpufree-data/venvs/mjlab/bin/python tools/monitor_tb.py $V > $H/runs/night/monitor_$V.out 2>&1 < /dev/null &
sleep 25
pgrep -af "tools/monitor_tb.py $V"
