#!/bin/bash
# Stop the remaining v5 seed-3 jobs (v6 supersedes v5; v5 stays an ablation step with seeds 1-2 complete) and
# release the v6 pipeline, which waits for "finished" in v5_seeds.log.  Finished seed-3 checkpoints are kept.
H=/root/gpufree-data/hnu_v2
pkill -f 'run_v5_seed[s].sh'
sleep 1
pkill -f 'distill_train.py --profile v5_Z'
pkill -f 'distill_train.py --profile v5d2_Z'
pkill -f 'eval_student.py --student /root/gpufree-data/hnu_v2/runs/students_v5_'
sleep 3
echo "[$(date +%H:%M:%S)] finished (seed 3 stopped early: superseded by v6)" >> $H/runs/night/v5_seeds.log
ps -eo pid,args | grep -e 'distill_trai[n]' -e 'train_hn[u]' -e 'run_v[6]' | cut -c1-120
