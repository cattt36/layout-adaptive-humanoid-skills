#!/bin/bash
# Restart the queued post-analysis of v6g10s with the corrected script (it only waits so far).
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "bash run_after_v6r.sh"); do kill $p; done
sleep 1
cd $H/code
V=v6g10s setsid nohup bash run_after_v6r.sh > $H/runs/night/after_v6g10s.out 2>&1 < /dev/null &
sleep 2
pgrep -af "run_after_v6r.sh"