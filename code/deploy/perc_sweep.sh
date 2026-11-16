#!/bin/bash
# Perception-error tolerance of an exported student (sim2sim, 0-20 ms latency, friction / CoM / encoder randomisation):
# constant obstacle offset 0 / 2 / 5 / 8 / 12 cm (random horizontal direction per episode) and per-step noise
# +-2 / 5 / 10 cm on the obstacle positions.
EX=$1
W=${2:-48}
PY=/root/gpufree-data/venvs/deploy/bin/python
cd /root/gpufree-data/hnu_v2/code
O=$EX/perception; mkdir -p $O
$PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --out $O/bias0.json
for b in 0.02 0.05 0.08 0.12; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --perc-bias $b --out $O/bias$b.json
done
for n in 0.02 0.05 0.10; do
  $PY deploy/sim2sim.py $EX --latency 0-20 --trials 4 --workers $W --perc-noise $n --out $O/noise$n.json
done
$PY deploy/s2s_report.py $O/bias0.json $O/bias0.0*.json $O/bias0.1*.json $O/noise*.json > $O/summary.txt
