#!/bin/bash
# The v4d chain fell back to sequential retries after Z1-Z3 failed at start-up (GPU memory at that moment).
# Stop the chain, run Z1/Z2/Z3 (RL seed 1) in parallel next to the running Z0, then restart the chain.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
pkill -f 'run_v4[d].sh'
sleep 2
echo "[$(date +%H:%M:%S)] parallel fix: Z1 Z2 Z3 (seed 1)" >> $N/v4d.log
bash tools/rerun_student.sh v4d Z1 1
bash tools/rerun_student.sh v4d Z2 1
bash tools/rerun_student.sh v4d Z3 0
for i in $(seq 1 400); do
  n=0; for P in Z0 Z1 Z2 Z3 Z4; do [ -f $H/runs/students_v4d/$P/student_2500.pt ] && n=$((n + 1)); done
  [ $n = 5 ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] all seed-1 students done, restarting chain" >> $N/v4d.log
nohup bash run_v4d.sh > $N/v4d_launcher2.log 2>&1 < /dev/null &
