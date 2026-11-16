#!/bin/bash
# Localisation-drift tolerance (no external tracking): obstacle layout known relative to the start pose up to an
# initial offset; robot pose from odometry with systematic scale / heading error r.  sim2sim, 0-20 ms latency,
# friction / CoM / encoder randomisation.
EX=$1
W=${2:-48}
PY=/root/gpufree-data/venvs/deploy/bin/python
cd /root/gpufree-data/hnu_v2/code
O=$EX/drift; mkdir -p $O
for r in 0.01 0.02 0.05 0.10; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --drift $r --out $O/r$r.json
done
for b in 0.02 0.05; do
  for r in 0.02 0.05; do
    $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --drift $r --init-off $b --out $O/b${b}_r$r.json
  done
done
$PY deploy/s2s_report.py $O/r*.json $O/b*.json > $O/summary.txt