#!/bin/bash
# Re-queue v6g10t with the task criterion as primary metric (tracking criterion kept as secondary).
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(pgrep -f "bash run_v6g10t.sh"); do kill $p; done
sleep 1
for DRV in 0 1; do  # the existing teacher evaluation used the tracking criterion -> secondary folder
  if [ -d $H/runs/results_v6g10t_bank_teacher_dr$DRV ] && [ ! -d $H/runs/results_v6g10ttrk_bank_teacher_dr$DRV ]; then
    mv $H/runs/results_v6g10t_bank_teacher_dr$DRV $H/runs/results_v6g10ttrk_bank_teacher_dr$DRV
  fi
done
echo "[$(date +%H:%M:%S)] re-queued: task criterion primary (results_v6g10t_*), tracking secondary (results_v6g10ttrk_*)" >> $H/runs/night/v6g10t.log
bash -n run_v6g10t.sh || exit 1
setsid nohup bash run_v6g10t.sh > /dev/null 2>&1 < /dev/null &
sleep 3
pgrep -af "run_v6g10t|run_latent_t" | cut -c1-80
ls -d $H/runs/results_v6g10t*bank* 2>/dev/null
tail -n 3 $H/runs/night/v6g10t.log
cat $H/runs/night/latent_v5t.log
