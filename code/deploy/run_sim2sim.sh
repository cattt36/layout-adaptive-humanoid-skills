#!/bin/bash
# Full sim2sim matrix for one exported student: nominal / latency / latency + DR, with and without pushes, plus a
# finer physics step (1 ms, closer to the motor drivers).  CPU only.
#   bash deploy/run_sim2sim.sh <export_dir> [workers]
EX=$1
W=${2:-48}
PY=/root/gpufree-data/venvs/deploy/bin/python
cd /root/gpufree-data/hnu_v2/code
$PY deploy/sim2sim.py $EX --trials 4 --workers $W
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --latency 0-20
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --latency 0-20 --dr
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --push
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --latency 0-20 --dr --push
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --physics-dt 0.001
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --physics-dt 0.001 --latency 0-20 --dr
$PY deploy/sim2sim.py $EX --trials 4 --workers $W --no-base-rand
$PY deploy/s2s_report.py $EX/sim2sim_*.json > $EX/sim2sim_summary.txt
