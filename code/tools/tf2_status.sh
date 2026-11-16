#!/bin/bash
# Progress of the v2 teacher fine-tunes: last iteration, stake / bridge terms, terminations, episode length.
TAG=${1:-pilot}
L=/root/gpufree-data/hnu_v2/runs/logs
for T in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do
  f=$L/${T}_tf2_$TAG.log
  [ -f $f ] || continue
  echo "== $T"
  grep -E 'Learning iteration' $f | tail -1
  grep -E 'Mean episode length|Episode_Reward/stake_event|Episode_Reward/feet_slip|Episode_Reward/bridge_contact|Episode_Reward/motion_body_pos|Episode_Termination/' $f | tail -12
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
