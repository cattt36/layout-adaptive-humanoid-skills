#!/bin/bash
# Restart v6g10n on the corrected fold-1 z tables (checkpoint normalizers).  Stops watchdog, pipeline, its running
# students and the monitor; the teacher-bank results (do not depend on z) are kept and skipped; no student had
# finished.  Then starts pipeline, monitor and watchdog again.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "tools/watchdog.sh v6g10n"); do kill $p; done
for p in $(pgrep -f "run_v6g10n.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10n|profile v6g10n|results_v6g10n_s"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10n"); do kill $p; done
sleep 10
ls $H/runs/students_v6g10n_*/*/student_4000.pt 2>/dev/null | head -3 || true
# partial students trained on the old z (only student_250.pt): moved aside (not deleted) so the restart starts clean
for d in $H/runs/students_v6g10n_s* $H/runs/logs/rsl_rl/students_v6g10n_s*; do
  case $d in *_oldz) ;; *) [ -e "$d" ] && mv "$d" "${d}_oldz" ;; esac
done
echo "[$(date +%H:%M:%S)] RESTART: z tables -> data/z_tables_cv_f1 (encoder checkpoint normalizers), training set aug_ts_f1; teacher-bank results kept" >> $N/v6g10n.log
grep -q "z tables exported" $N/cv_prep.log 2>/dev/null; true
cd $H/code
setsid nohup bash run_v6g10n.sh >> $N/run_v6g10n.out 2>&1 < /dev/null &
sleep 10
bash tools/restart_monitor.sh v6g10n
setsid nohup bash tools/watchdog.sh v6g10n > /dev/null 2>&1 < /dev/null &
sleep 60
tail -4 $N/v6g10n.log
pgrep -af "distill_train.py --profile v6g10n" | cut -c1-140
