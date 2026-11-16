#!/bin/bash
# Post-analyses (items 1-5) for the bridge-scene students (v6r2) and the final unified-scene students (v6g10).
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(ps -eo pid,args | grep 'run_after_v6[r].sh' | awk '{print $1}'); do kill $p; done
V=v6r2 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
V=v6g10 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 2
ps -eo args | grep -c 'run_after_v6[r].sh'
