#!/bin/bash
# Generic server-side watchdog of a student batch V (default v6g10n): every 30 min
#   * pipeline dead, batch not finished and no training / eval process of V left -> restart run_$V.sh (max 5)
#   * TensorBoard monitor dead -> restart it
#   * status snapshot every 2 h -> runs/night/watchdog_$V.txt
# Deletes nothing (restart_monitor.sh recreates only the monitor's own TensorBoard run dir).
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=${1:-v6g10n}
S=${2:-run_$V.sh}  # pipeline script (L40 server: run_v6g10n_l40.sh)
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
  if ! pgrep -f "$S" > /dev/null && ! pgrep -f "students_${V}_|results_${V}_|profile ${V}" > /dev/null; then
    if [ $restarts -lt 5 ]; then
      restarts=$((restarts + 1))
      echo "[$(date '+%m-%d %H:%M')] pipeline dead -> restart $restarts" >> $W
      tail -20 $N/${S%.sh}.out >> $W 2>&1
      setsid nohup bash $S >> $N/${S%.sh}.out 2>&1 < /dev/null &
    else
      echo "[$(date '+%m-%d %H:%M')] pipeline dead, restart limit reached" >> $W
    fi
  fi
  if [ "$S" = "run_$V.sh" ] && ! pgrep -f "tools/monitor_tb.py $V" > /dev/null; then
    echo "[$(date '+%m-%d %H:%M')] monitor dead -> restart" >> $W
    bash tools/restart_monitor.sh $V >> $W 2>&1
  fi
  [ $((i % 4)) -eq 0 ] && bash tools/status_m.sh $V >> $W 2>&1
  i=$((i + 1))
  sleep 1800
done
