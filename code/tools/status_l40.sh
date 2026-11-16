#!/bin/bash
# Status of the L40 server: queue, running students (iteration), GPUs, errors, helpers.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
date
echo "== log"; tail -4 $N/v6g10n.log
echo "== running students"
for p in $(pgrep -f "distill_train.py --profile"); do
  prof=$(tr '\0' ' ' < /proc/$p/cmdline | sed 's/.*--profile \([^ ]*\).*/\1/')
  out=$(tr '\0' ' ' < /proc/$p/cmdline | sed 's/.*--out \([^ ]*\).*/\1/')
  last=$(tail -1 $out/train_log.jsonl 2>/dev/null | sed 's/.*"it": \([0-9]*\).*"elapsed_min": \([0-9.]*\).*/it \1, \2 min/')
  echo "  $prof  $last"
done
echo "  evaluations running: $(pgrep -fc 'eval_student.py --student')"
echo "== done: round1 $(ls $H/runs/students_v6g10n*_s*/*/student_4000.pt 2>/dev/null | grep -v _d2_ | grep -vc paused)  round2 $(ls $H/runs/students_v6g10n*_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)  claimed jobs $(ls $N/v6g10n_l40_claims | wc -l)/76"
echo "== folds ready: $(ls $H/data/z_tables_cv_f*/ready 2>/dev/null | sed 's/.*cv_f\([0-9]\).*/\1/' | tr '\n' ' ')"
echo "== gpus"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "== errors (last 3 h)"; find $N -name "*.log" -mmin -180 | xargs grep -l -i "traceback\|error" 2>/dev/null | head -5
echo "== helpers"; ps -eo args | grep -E "watchdog.sh|monitor_tb.py|tensorboard --logdir|run_v6g10n_l40" | grep -v grep | cut -c1-70 | sort | uniq -c
df -h /root/gpufree-data | tail -1
