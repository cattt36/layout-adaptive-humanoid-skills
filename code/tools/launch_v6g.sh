#!/bin/bash
# (Re)launch v6g and its post-analysis (V=v6g); stop the old post-analysis waiting for v6r2.
H=/root/gpufree-data/hnu_v2
cd $H/code
pkill -f 'run_after_v6[r].sh'
pkill -f 'run_v6[g].sh'
sleep 1
setsid nohup bash run_v6g.sh > /dev/null 2>&1 < /dev/null &
V=v6g setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 3
ps -eo args | grep -e 'run_v6[g]' -e 'run_after_v6[r]' -e 'robust_stake[s]' -e 'run_v6r[2]' | sort | uniq -c
cat $H/runs/night/v6g.log
