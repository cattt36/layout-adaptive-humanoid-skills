#!/bin/bash
# Stop the v6g10m batch (superseded by the teacher fine-tune + redistillation): watchdog first (it
# would restart the pipeline), then the pipeline, its jobs and the monitor.  Nothing is deleted.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "tools/watchdog_m.sh"); do kill $p; done
for p in $(pgrep -f "run_v6g10m.sh"); do kill $p; done
for p in $(pgrep -f "tools/reeval_v2.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10m|results_v6g10m|profile v6g10m"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10m"); do kill $p; done
sleep 10
echo "[$(date +%H:%M:%S)] stopped by user decision (teacher fine-tune with contact-based stake rewards)" >> $H/runs/night/v6g10m.log
echo "left over:"; pgrep -af "v6g10m|watchdog_m|reeval_v2" | grep -v stop_m || echo none
grep -c "done" $H/runs/night/v6g10m.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
