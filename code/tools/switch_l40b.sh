#!/bin/bash
# L40 server, (user decision): switch the student pipeline from run_v6g10n_l40.sh to run_v6g10n_l40b.sh at
# the batch boundary (diagnostic jobs first, same recipe).  Nothing running is interrupted:
#   1. stop the watchdog of the old scheduler;
#   2. close the old job queue (claim every job index, so the old workers take no new job);
#   3. stop only the old parent shell (it would otherwise run its sequential retry pass) -- the worker shells finish
#      their current student incl. its evaluation and then exit;
#   4. when no old worker is left: start run_v6g10n_l40b.sh (its claim directory is reset, finished parts are skipped)
#      and a new watchdog for it.
# Log: runs/night/v6g10n.log
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
Q=$N/v6g10n_l40_claims
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/v6g10n.log; }
pkill -f "tools/watchdog.sh v6g10n run_v6g10n_l40.sh"
for i in $(seq 0 75); do mkdir -p $Q/$i; done
P=$(pgrep -o -f "bash run_v6g10n_l40.sh")
if [ -n "$P" ] && [ "$(ps -o ppid= -p $P | tr -d ' ')" = 1 ]; then kill $P; fi
log "switch: old queue closed, old parent $P stopped; the running students finish, then run_v6g10n_l40b.sh starts"
while pgrep -f "bash run_v6g10n_l40.sh" > /dev/null; do sleep 30; done
log "switch: old workers finished -> start run_v6g10n_l40b.sh"
setsid nohup bash run_v6g10n_l40b.sh >> $N/run_v6g10n_l40b.out 2>&1 < /dev/null &
sleep 10
setsid nohup bash tools/watchdog.sh v6g10n run_v6g10n_l40b.sh >> $N/watchdog_v6g10n.out 2>&1 < /dev/null &
log "switch: done (watchdog for run_v6g10n_l40b.sh started)"
