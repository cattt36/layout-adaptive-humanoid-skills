#!/bin/bash
# Quick status of running jobs.
H=/root/gpufree-data/hnu_v2
date +%H:%M:%S
tail -n 4 $H/runs/night/v4d.log
ps -eo pid,etime,args | grep -E 'v4d_paralle[l]|run_v4[d]|distill_trai[n]|train_hn[u]|eval_[e]xpert|eval_[s]tudent' | cut -c1-150
for p in Z0 Z1 Z2 Z3 Z4; do
  echo "$p $(ls $H/runs/students_v4d/$p 2>/dev/null | grep student_ | sort -V | tail -n 1) $(tail -n 1 $H/runs/students_v4d/$p/train_log.jsonl 2>/dev/null | cut -c1-60)"
done
for f in $H/runs/logs/*stakes3*.log; do echo "$f $(grep -a 'Learning iteration' $f | tail -n 1)"; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
