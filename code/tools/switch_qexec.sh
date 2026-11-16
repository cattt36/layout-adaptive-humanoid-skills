#!/bin/bash
# Stop finish_A.sh (its Q_exec part runs out of the 100 GB container memory) and start q_exec_A.sh.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "bash finish_A.sh"); do kill $p; done
for p in $(pgrep -f "stage5_quality/quality.py"); do kill $p; done
sleep 3
cd $H/code
setsid nohup bash q_exec_A.sh > $H/runs/night/q_exec_A.out 2>&1 < /dev/null &
sleep 2
pgrep -af "q_exec_A.sh|quality.py" | cut -c1-120
