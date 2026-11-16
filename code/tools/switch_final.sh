#!/bin/bash
# User all three actions use the 0-20 ms latency model; stop the 0-40 ms bridge-only batch (kept as the
# latency-range ablation) and go straight to the unified-scene students once the stake teachers pass.
H=/root/gpufree-data/hnu_v2
for p in $(ps -eo pid,args | grep -e 'run_bridge_students_r[2].sh' -e 'retrain_seed[1].sh' -e 'move_d[3].sh' -e 'run_after_v6[r].sh' -e 'run_v6g1[0].sh' | awk '{print $1}'); do kill $p; done
pkill -f 'distill_train.py --profile v6r2'
pkill -f 'eval_student.py --student /root/gpufree-data/hnu_v2/runs/students_v6r2'
sleep 3
echo "[$(date +%H:%M:%S)] stopped: bridge-only 0-40 ms batch kept as latency-range ablation (user: 0-20 ms for all actions)" >> $H/runs/night/v6r2.log
cd $H/code
setsid nohup bash run_v6g10.sh > /dev/null 2>&1 < /dev/null &
V=v6g10 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 3
ps -eo args | grep -e 'run_v6g1[0]' -e 'run_after_v6[r]' -e 'robust_stakes_h10[c]' | sort | uniq -c
bash tools/gpu_procs.sh
