#!/bin/bash
# Inspect the v4d H stage (offline 8.1 + language) logs and the stakes eval waiter.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for S in 42 43 44; do echo "== H_$S"; tail -n 6 $N/v4d_H_$S.log; done
echo "== offline_81_all"; tail -n 25 $N/offline_81_all.md
echo "== language_all"; tail -n 30 $N/language_all.md
echo "== stakes waiter"; ps -eo pid,etime,args | grep -E 'eval_stake[s]3|eval_exper[t]' | cut -c1-120
grep -ac 'Learning iteration 17997/17998' $H/runs/logs/ThreeStepPoleRun_stakes3.log $H/runs/logs/FiveStepPoleRun_stakes3.log
