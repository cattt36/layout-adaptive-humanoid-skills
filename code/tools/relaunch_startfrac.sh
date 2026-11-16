#!/bin/bash
# Fine-tuned experts (stakes, robust) forget the start of the motion: failures at 9-22 % of the clip although the
# training episodes are long (adaptive sampling rarely starts at frame 0 any more).  Relaunch both lines with 25 %
# of the resets at the first frame of a clip.
H=/root/gpufree-data/hnu_v2
pkill -f 'launch_stakes_gat[e].sh'
pkill -f 'run_v6r[2].sh'
pkill -f 'run_after_v6[r].sh'
sleep 1
pkill -f 'run-name gate_'
pkill -f 'run-name r2_'
sleep 5
rm -rf $H/runs/logs/rsl_rl/hnu_track_*/*_gate_flat $H/runs/logs/rsl_rl/hnu_track_*/*_gate_h06 $H/runs/logs/rsl_rl/hnu_track_*/*_pad_flat
rm -rf $H/runs/logs/rsl_rl/hnu_track_*/*_r2_*
rm -f $H/runs/night/v6r2_*.done
echo "[$(date +%H:%M:%S)] restart with HNU_START_FRAC=0.25 (start-of-motion forgetting)" >> $H/runs/night/v6r2.log
cd $H/code
export HNU_START_FRAC=0.25
setsid nohup bash tools/launch_stakes_gate.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash run_v6r2.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash run_after_v6r.sh > /dev/null 2>&1 < /dev/null &
sleep 3
ps -eo args | grep -c 'train_hn[u]'
