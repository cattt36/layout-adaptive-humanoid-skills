#!/bin/bash
# Restart the stakes3 evaluation waiter (the last logged iteration is 17997/17998, not 17998/17998).
H=/root/gpufree-data/hnu_v2
cd $H/code
pkill -f 'eval_stake[s]3.sh'
sed -i 's#Learning iteration 17998/17998#Learning iteration 17997/17998#' tools/eval_stakes3.sh
nohup bash tools/eval_stakes3.sh stakes3 > /dev/null 2>&1 < /dev/null &
sleep 1
ps -eo pid,args | grep 'eval_stake[s]3'
ls $H/runs/results_v4d
tail -n 5 $H/runs/night/v4d_F_Z0_s1.log
