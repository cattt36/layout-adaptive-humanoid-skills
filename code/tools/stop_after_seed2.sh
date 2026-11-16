#!/bin/bash
# Keep 2 RL seeds for v3 (stop run_seeds.sh once seed 2 is evaluated) so that the v4b chain can start earlier.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
while ! grep -q 'seed 3 train' $N/seeds.log 2>/dev/null; do sleep 30; done
pkill -f 'run_seed[s].sh'
sleep 2
for p in $(pgrep -f 'students_v3_s[3]'); do kill $p; done
$PY tools/aggregate_seeds.py $H/runs/results_v3 $H/runs/results_v3_s2 > $N/tables_v3_seeds.md 2>&1
echo "[$(date +%H:%M:%S)] seed 3 cancelled (priority to v4b); seeds done" >> $N/seeds.log
