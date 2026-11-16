#!/bin/bash
# Status of the trial-level batch: latents, router, 8.1, students.
N=/root/gpufree-data/hnu_v2/runs/night
date
echo "== latents v5t"; tail -n 6 $N/latent_v5t.log
echo "== students v6g10t"; tail -n 4 $N/v6g10t.log | cut -c1-140
for f in $N/v6g10t_s*_E_*.log $N/v6g10t_d2_s*_*.log; do [ -f $f ] && echo "  $(basename $f) $(tail -n 1 $f | cut -c1-48)"; done
echo "== router v5t"; grep -v -i warn $N/router_ft2_v5t.log 2>/dev/null | tail -n 7
grep -v -i warn $N/router_test_v2_v5t.md 2>/dev/null | tail -n 8
echo "== 8.1 v5t"; grep -A 24 "latent_v5t (3 seeds)" $N/offline81_summary_all.md 2>/dev/null
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
