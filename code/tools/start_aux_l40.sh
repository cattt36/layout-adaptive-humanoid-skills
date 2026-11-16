#!/bin/bash
# L40 server: watchdog of run_v6g10n_l40.sh, TensorBoard monitors (fold 1 = monitor_v6g10n, folds 2-4 = monitor_v6g10n_f<k>)
# and TensorBoard on 127.0.0.1:6017 (view through an ssh tunnel).
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
pgrep -f "tools/watchdog.sh v6g10n run_v6g10n_l40.sh" > /dev/null || \
  setsid nohup bash tools/watchdog.sh v6g10n run_v6g10n_l40.sh > /dev/null 2>&1 < /dev/null &
for F in 1 2 3 4; do
  pgrep -f "tools/monitor_tb.py v6g10n $F" > /dev/null || \
    CUDA_VISIBLE_DEVICES= setsid nohup $PY tools/monitor_tb.py v6g10n $F > $H/runs/night/monitor_v6g10n_f$F.out 2>&1 < /dev/null &
done
pgrep -f "tensorboard --logdir" > /dev/null || \
  setsid nohup $PY /root/gpufree-data/venvs/mjlab/bin/tensorboard --logdir $H/runs/logs/rsl_rl --host 127.0.0.1 --port 6017 \
    --reload_interval 30 --samples_per_plugin text=100 > $H/runs/night/tensorboard.out 2>&1 < /dev/null &
sleep 40
ps -eo pid,args | grep -E "watchdog|monitor_tb|tensorboard --logdir" | grep -v grep | cut -c1-110
tail -3 $H/runs/night/monitor_v6g10n_f1.out $H/runs/night/monitor_v6g10n_f2.out
