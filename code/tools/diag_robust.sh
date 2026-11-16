#!/bin/bash
# Stop the post-v6r analyses (v6r students died with OOM) and look at the robust-teacher training curves.
H=/root/gpufree-data/hnu_v2
pkill -f 'run_after_v6[r].sh'
pkill -f 'quality.py collect --policy student --student /root/gpufree-data/hnu_v2/runs/students_v6r'
echo "[$(date +%H:%M:%S)] NOTE: students failed (CUDA OOM, GPU shared with stake training); robust teachers weak -> v6r2" >> $H/runs/night/v6r.log
for f in CrouchedTraverse_robust_bridge ThreeStepPoleRun_robust_flat FiveStepPoleRun_robust_flat; do
  echo "== $f"
  grep -a -e 'Learning iteration' -e 'Mean episode length' -e 'Mean reward' $H/runs/logs/$f.log | tr -s ' ' | \
    awk 'NR%3==1{it=$0} NR%3==2{ep=$0} NR%3==0{print it, ep, $0}' | sed -n '1p;20p;60p;150p;300p;600p;$p'
  grep -a -o 'Episode_Termination/[a-z_]*: [0-9.]*' $H/runs/logs/$f.log | tail -n 7
done
for T in ThreeStepPoleRun FiveStepPoleRun; do echo "$T $(grep -a 'Learning iteration' $H/runs/logs/${T}_runup_h06.log | tail -n 1 | tr -s ' ')"; done
