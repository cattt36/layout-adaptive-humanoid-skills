#!/bin/bash
# After an instance restart: held-out pipeline, TensorBoard (port 6017) and its monitor.
H=/root/gpufree-data/hnu_v2
cd $H/code
echo "[$(date +%H:%M:%S)] instance restarted (GPU 0 freed); relaunch" >> $H/runs/night/v6g10h.log
setsid nohup bash run_v6g10h.sh > /dev/null 2>&1 < /dev/null &
setsid nohup /root/gpufree-data/venvs/mjlab/bin/tensorboard --logdir $H/runs/logs/rsl_rl --host 127.0.0.1 --port 6017 \
  --reload_interval 30 --samples_per_plugin text=100 > $H/runs/night/tensorboard_6017.out 2>&1 < /dev/null &
sleep 5
bash tools/restart_monitor.sh v6g10h > /dev/null 2>&1
sleep 90
tail -n 2 $H/runs/night/v6g10h.log
pgrep -af "run_v6g10h.sh" | head -n 1
pgrep -af "distill_train.py" | cut -c1-90
pgrep -af "port 6017" | cut -c1-90
pgrep -af "monitor_tb.py" | cut -c1-90
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
