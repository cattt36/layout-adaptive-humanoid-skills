#!/bin/bash
# Tolerance to misplaced obstacles (true layout differs from the reference clip, perception correct), sim2sim,
# 0-20 ms latency, friction / CoM / encoder randomisation.
EX=$1
W=${2:-48}
PY=/root/gpufree-data/venvs/deploy/bin/python
cd /root/gpufree-data/hnu_v2/code
O=$EX/placement; mkdir -p $O
for b in 0.02 0.05 0.08 0.12; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --obst-shift $b --out $O/shift$b.json
done
for j in 0.02 0.05 0.08; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --tasks ThreeStepPoleRun,FiveStepPoleRun --stake-jitter $j --out $O/jitter$j.json
done
$PY deploy/s2s_report.py $O/shift*.json $O/jitter*.json > $O/summary.txt