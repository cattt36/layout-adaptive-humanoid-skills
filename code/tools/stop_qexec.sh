#!/bin/bash
# Stop the item-5 Q_exec run on the old all-clip students (to be redone on the held-out students) and release the
# held-out distillation, which waits for "finished" in q_exec_A.log.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "bash q_exec_A.sh"); do kill $p; done
for p in $(pgrep -f "stage5_quality/quality.py"); do kill $p; done
sleep 3
echo "[$(date +%H:%M:%S)] stopped early by request (finished = released): Q_exec is redone on the held-out students" >> $H/runs/night/q_exec_A.log
echo "[$(date +%H:%M:%S)] finished" >> $H/runs/night/q_exec_A.log
pgrep -af "q_exec_A|quality.py" | cut -c1-100
