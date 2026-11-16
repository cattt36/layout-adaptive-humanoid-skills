#!/bin/bash
# Stop the USD-with-posts bridge runs (episodes collapse to ~45 steps: reference states inside the posts).
pkill -f 'bridge_usd_pipeline.s[h]'
pkill -f 'run-name bridge_usd_(faithfu[l]|lif[t])'
sleep 3
ps -eo pid,args | grep -c 'train_hn[u]'
