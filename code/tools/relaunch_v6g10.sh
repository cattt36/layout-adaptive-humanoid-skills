#!/bin/bash
cd /root/gpufree-data/hnu_v2/code
echo "[$(date +%H:%M:%S)] user approved h10c stake teachers (robot-scale-stride retention ok: W1 100->99, W2 87.5->90.6); relaunch" >> ../runs/night/v6g10.log
setsid nohup bash run_v6g10.sh > /dev/null 2>&1 < /dev/null &