#!/bin/bash
# Queue items 1-3: replace run_ref.sh by the joint REF + seeds-4/5 queue, then the 32-trial evaluation, then the
# single-subject low-data batch; monitors for v6g10h (5 seeds) and v6g10q (test split).
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(pgrep -f "bash run_ref.sh"); do kill $p; done
echo "[$(date +%H:%M:%S)] cancelled: REF students moved into run_v6g10h45.sh (one free-memory queue)" >> $H/runs/night/v6g10h_ref.log
for f in run_v6g10h45.sh run_eval32.sh run_v6g10q.sh; do bash -n $f || exit 1; setsid nohup bash $f > /dev/null 2>&1 < /dev/null & done
sleep 3
bash tools/restart_monitor.sh v6g10h > /dev/null 2>&1
for p in $(pgrep -f "tools/monitor_tb.py v6g10q"); do kill $p; done
MQ=$H/runs/logs/rsl_rl/monitor_v6g10q
[ -d $MQ ] && rm -r $MQ
[ -d ${MQ}_detail ] && rm -r ${MQ}_detail
HNU_MON_SPLIT=test CUDA_VISIBLE_DEVICES= setsid nohup /root/gpufree-data/venvs/mjlab/bin/python tools/monitor_tb.py v6g10q > $H/runs/night/monitor_v6g10q.out 2>&1 < /dev/null &
sleep 5
tail -n 1 $H/runs/night/v6g10h_s45.log $H/runs/night/v6g10h32.log $H/runs/night/v6g10q.log
pgrep -af "run_v6g10h.sh|run_v6g10h45|run_eval32|run_v6g10q|run_ref|monitor_tb" | cut -c1-90
