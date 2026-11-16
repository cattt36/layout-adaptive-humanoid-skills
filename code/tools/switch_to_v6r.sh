#!/bin/bash
# Replace the nominal v6 run by v6 + robustness (run_v6r.sh), which starts after the 0.06 m stake experts finish.
H=/root/gpufree-data/hnu_v2
pkill -f 'run_v[6].sh'
sleep 1
pkill -f 'distill_train.py --profile v6_Z'
pkill -f 'distill_train.py --profile v6d2_Z'
sleep 3
echo "[$(date +%H:%M:%S)] stopped: replaced by v6r (v6 + robustness)" >> $H/runs/night/v6.log
cd $H/code
setsid nohup bash run_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 2
ps -eo pid,args | grep -e 'distill_trai[n]' -e 'train_hn[u]' -e 'run_v6[r]' | cut -c1-110
