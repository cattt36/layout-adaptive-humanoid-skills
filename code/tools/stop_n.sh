#!/bin/bash
# Pause v6g10n and the waiting CV-student scheduler on this server: all students (folds 1-4) move to the rented 4x L40
# server so that every student is trained and evaluated on the same hardware.  The CV latent preparation
# (tools/run_cv_prep.sh) keeps running here.  Nothing is deleted; partial students -> *_4090paused.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "tools/watchdog.sh v6g10n"); do kill $p; done
for p in $(pgrep -f "run_v6g10n.sh"); do kill $p; done
for p in $(pgrep -f "run_cv_students.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10n|profile v6g10n|results_v6g10n_s"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10n"); do kill $p; done
sleep 10
for d in $H/runs/students_v6g10n_s1 $H/runs/logs/rsl_rl/students_v6g10n_s1; do
  [ -e "$d" ] && mv "$d" "${d}_4090paused"
done
echo "[$(date +%H:%M:%S)] PAUSED: all students (folds 1-4) move to the 4x L40 server (same hardware for every student); CV latent prep continues here" >> $N/v6g10n.log
echo "[$(date '+%m-%d %H:%M:%S')] scheduler stopped here: CV students run on the 4x L40 server" >> $N/cv_students.log
ps -eo pid,args | grep -E "distill_train|run_v6g10n|run_cv_students|watchdog" | grep -v grep | cut -c1-100
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
