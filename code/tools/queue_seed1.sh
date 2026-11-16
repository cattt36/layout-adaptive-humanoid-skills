#!/bin/bash
# Queue order (user ): bridge seeds 2-3 -> seed 1 retrain -> (robust 0.10 m stake teachers in parallel)
# -> unified-scene students (v6g10) -> post-analyses.
H=/root/gpufree-data/hnu_v2
cd $H/code
for p in $(ps -eo pid,args | grep -e 'run_v6g1[0].sh' -e 'run_after_v6[r].sh' | awk '{print $1}'); do kill $p; done
sleep 1
setsid nohup bash tools/retrain_seed1.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash run_v6g10.sh > /dev/null 2>&1 < /dev/null &
WAIT_MARK="seed1 retrain done" V=v6r2 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
V=v6g10 setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 2
ps -eo args | grep -e 'retrain_seed[1]' -e 'run_v6g1[0]' -e 'run_after_v6[r]' -e 'run_bridge_students_r[2]' -e 'robust_stakes_h10[b]' | sort | uniq -c
