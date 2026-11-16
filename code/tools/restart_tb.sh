#!/bin/bash
# Restart the TensorBoard on port 6017 (logdir runs/logs/rsl_rl): drops stale data of deleted / rewritten runs.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "tensorboard --logdir $H/runs/logs/rsl_rl --host 127.0.0.1 --port 6017"); do kill $p; done
sleep 3
setsid nohup /root/gpufree-data/venvs/mjlab/bin/tensorboard --logdir $H/runs/logs/rsl_rl --host 127.0.0.1 --port 6017 \
  --reload_interval 30 --samples_per_plugin text=100 > $H/runs/night/tensorboard_6017.out 2>&1 < /dev/null &
sleep 15
pgrep -af "port 6017"
ss -ltn | grep 6017
