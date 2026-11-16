#!/bin/bash
# Cancel the subject-held-out queues (seeds 4-5 / REF, 32-trial evaluation, low-data) and their monitors; start the
# trial-level held-out batch v6g10t and its monitor.
H=/root/gpufree-data/hnu_v2
cd $H/code
for s in run_v6g10h45.sh run_eval32.sh run_v6g10q.sh; do
  for p in $(pgrep -f "bash $s"); do kill $p; done
done
for f in v6g10h_s45 v6g10h32 v6g10q; do
  echo "[$(date +%H:%M:%S)] cancelled: replaced by the trial-level protocol (run_v6g10t.sh)" >> $H/runs/night/$f.log
done
for v in v6g10h v6g10q; do for p in $(pgrep -f "tools/monitor_tb.py $v"); do kill $p; done; done
bash -n run_v6g10t.sh || exit 1
setsid nohup bash run_v6g10t.sh > /dev/null 2>&1 < /dev/null &
sleep 3
bash tools/restart_monitor.sh v6g10t > /dev/null 2>&1
sleep 5
pgrep -af "run_v6g10|run_eval32|monitor_tb" | cut -c1-90
cat $H/runs/night/v6g10t.log
