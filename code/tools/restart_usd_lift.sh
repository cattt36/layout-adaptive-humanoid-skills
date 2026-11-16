#!/bin/bash
# Restart the usd_lift variant with the iterated plank lift.
H=/root/gpufree-data/hnu_v2
pkill -f 'bridge_usd_pipeline.sh usd_lif[t]'
pkill -f 'run-name bridge_usd_lif[t]'
pkill -f 'ref_bridge_usd_lif[t]'
sleep 3
rm -rf $H/data/bridge_usd_lift $H/data/ref_bridge_usd_lift $H/runs/bridge_eval/usd_lift_zeroshot_bridge1
rm -rf $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_bridge_usd_lift
cd $H/code
setsid nohup bash tools/bridge_usd_pipeline.sh usd_lift 1 6000 > /dev/null 2>&1 < /dev/null &
echo restarted
