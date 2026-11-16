#!/bin/bash
# Restart v6r2 with the retention gate (script changed; bash must not keep executing the old file).
H=/root/gpufree-data/hnu_v2
pkill -f 'run_v6r[2].sh'
sleep 1
pkill -f 'run-name r2_'
sleep 5
rm -rf $H/runs/logs/rsl_rl/hnu_track_*/*_r2_*
echo "[$(date +%H:%M:%S)] restart: retention gate added (experts must keep their original motion)" >> $H/runs/night/v6r2.log
cd $H/code
export HNU_START_FRAC=0.25
setsid nohup bash run_v6r2.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash tools/retention_stakes.sh > /dev/null 2>&1 < /dev/null &
sleep 60
bash tools/running.sh
