#!/bin/bash
# Server-side watchdog of v6g10m while nobody supervises: every 30 min
#   * pipeline dead, batch not finished and no v6g10m training / eval process left -> restart run_v6g10m.sh
#     (finished students are skipped; at most 5 restarts)
#   * TensorBoard monitor dead -> restart it (tools/restart_monitor.sh)
#   * appends a status snapshot (tools/status_m.sh) to runs/night/watchdog_v6g10m.txt every 2 h
# Never deletes anything except the monitor's own TensorBoard run dir (done by restart_monitor.sh).
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10m
W=$N/watchdog_$V.txt
cd $H/code
restarts=0
i=0
while true; do
  if grep -q "finished" $N/$V.log 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M')] batch finished, watchdog exits" >> $W
    bash tools/status_m.sh $V >> $W 2>&1
    exit 0
  fi
  if ! pgrep -f "run_$V.sh" > /dev/null && ! pgrep -f "students_${V}_|results_${V}_|profile ${V}" > /dev/null; then
    if [ $restarts -lt 5 ]; then
      restarts=$((restarts + 1))
      echo "[$(date '+%m-%d %H:%M')] pipeline dead -> restart $restarts" >> $W
      tail -20 $N/run_$V.out >> $W 2>&1
      setsid nohup bash run_$V.sh >> $N/run_$V.out 2>&1 < /dev/null &
    else
      echo "[$(date '+%m-%d %H:%M')] pipeline dead, restart limit reached" >> $W
    fi
  fi
  if ! pgrep -f "tools/monitor_tb.py $V" > /dev/null; then
    echo "[$(date '+%m-%d %H:%M')] monitor dead -> restart" >> $W
    bash tools/restart_monitor.sh $V >> $W 2>&1
  fi
  [ $((i % 4)) -eq 0 ] && bash tools/status_m.sh $V >> $W 2>&1
  i=$((i + 1))
  sleep 1800
done
