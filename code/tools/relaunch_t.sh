#!/bin/bash
# Stop the student batch that used the subject-split latents (z_tables_v5), start the trial-split latents (v5t)
# and re-queue the student batch on the new z tables; restart its monitor.
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(pgrep -f "bash run_v6g10t.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "distill_train.py --profile v6g10t"); do kill $p; done
for p in $(pgrep -f "eval_student.py --student /root/gpufree-data/hnu_v2/runs/students_v6g10t"); do kill $p; done
sleep 5
echo "[$(date +%H:%M:%S)] stopped: the latent (z_tables_v5) had seen test trials; re-queued on trial-split latents v5t" >> $H/runs/night/v6g10t.log
bash -n run_latent_t.sh && bash -n run_v6g10t.sh || exit 1
setsid nohup bash run_latent_t.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash run_v6g10t.sh > /dev/null 2>&1 < /dev/null &
sleep 3
bash tools/restart_monitor.sh v6g10t > /dev/null 2>&1
sleep 120
cat $H/runs/night/latent_v5t.log
tail -n 2 $H/runs/night/v6g10t.log
ls $H/runs/latent_v5t/
tail -n 3 $H/runs/latent_v5t/train_Z0_seed42.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
