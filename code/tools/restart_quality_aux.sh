#!/bin/bash
# Restart the quality-aux runs from scratch (aux targets changed).
H=/root/gpufree-data/hnu_v2
pkill -f 'run_quality_au[x].sh'
pkill -f 'stage5_quality/quality.p[y] train'
sleep 2
rm -rf $H/runs/quality_aux_v4d
cd $H/code
setsid nohup bash tools/run_quality_aux.sh 1 > /dev/null 2>&1 < /dev/null &
sleep 90
cat $H/runs/quality_aux_v4d/aux.log | grep -E 'level|GRF'
