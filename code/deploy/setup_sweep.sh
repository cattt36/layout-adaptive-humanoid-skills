#!/bin/bash
# Set-up tolerances of an exported student (sim2sim, 0-20 ms latency, friction / CoM / encoder randomisation):
# bridge built 3 / 5 cm lower or higher than the reference (perceived correctly), robot start heading error 3 / 5 / 10 deg.
EX=$1
W=${2:-48}
PY=/root/gpufree-data/venvs/deploy/bin/python
cd /root/gpufree-data/hnu_v2/code
O=$EX/setup; mkdir -p $O
for dz in -0.05 -0.03 0.03 0.05; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --tasks CrouchedTraverse --bridge-dz $dz --out $O/bridge_dz$dz.json
done
for y in 3 5 10; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --start-yaw $y --out $O/yaw$y.json
done
$PY deploy/s2s_report.py $O/bridge_dz*.json $O/yaw*.json > $O/summary.txt
