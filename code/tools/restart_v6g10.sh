#!/bin/bash
# Restart the (still waiting) final-student pipeline so that it uses the latency-focused stake teachers (h10c).
H=/root/gpufree-data/hnu_v2
for p in $(ps -eo pid,args | grep 'run_v6g1[0].sh' | awk '{print $1}'); do kill $p; done
sleep 1
cd $H/code
setsid nohup bash run_v6g10.sh > /dev/null 2>&1 < /dev/null &
sleep 2
tail -n 1 $H/runs/night/v6g10.log
