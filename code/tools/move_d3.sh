#!/bin/bash
# When the seed-1 retrain starts, archive the latency fix-up models (d3) so that later steps use the retrained seed 1.
H=/root/gpufree-data/hnu_v2
for i in $(seq 1 3000); do grep -q 'seed 1 retrain from scratch' $H/runs/night/v6r2.log && break; sleep 60; done
[ -d $H/runs/students_v6r2_d3_s1 ] && mv $H/runs/students_v6r2_d3_s1 $H/runs/students_v6r2_d3_s1old
echo "[$(date +%H:%M:%S)] archived fix-up models -> students_v6r2_d3_s1old" >> $H/runs/night/v6r2.log
