#!/bin/bash
# Stop the v6g10 students (old latency model: independent per-term lags, action / z_skill delayed) and their queued
# post-analysis; start v6g10s (sync, sensor-only 0-20 ms latency) + its post-analysis.
H=/root/gpufree-data/hnu_v2
cd $H/code
kill 1377163 1378703 1378704 1138882 1380235 2>/dev/null
sleep 2
pkill -f 'distill_train.py --profile v6g10_' ; pkill -f 'distill_train.py --profile v6g10d2_' ; pkill -f 'eval_student.py --student /root/gpufree-data/hnu_v2/runs/students_v6g10_'
sleep 20
echo "[$(date +%H:%M:%S)] stopped: superseded by v6g10s (corrected latency model)" >> $H/runs/night/v6g10.log
setsid nohup bash run_v6g10s.sh > /dev/null 2>&1 < /dev/null &
V=v6g10s setsid nohup bash run_after_v6r.sh > $H/runs/night/after_v6g10s.out 2>&1 < /dev/null &
sleep 5
ps -eo pid,args | grep -e '[r]un_v6g10' -e '[r]un_after' -e '[d]istill_train' | cut -c1-120
