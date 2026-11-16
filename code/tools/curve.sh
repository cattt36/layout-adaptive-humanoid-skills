#!/bin/bash
# Training curve (iteration, episode length, reward) and last termination counts of rsl_rl logs.
H=/root/gpufree-data/hnu_v2
for f in "$@"; do
  L=$H/runs/logs/$f.log
  echo "== $f"
  paste <(grep -a 'Learning iteration' $L | tr -s ' ' | cut -d' ' -f4) \
        <(grep -a 'Mean episode length' $L | tr -s ' ' | cut -d' ' -f5) \
        <(grep -a 'Mean reward' $L | tr -s ' ' | cut -d' ' -f4) | awk 'NR==1||NR%800==0{print} END{print}'
  grep -a -o 'Episode_Termination/[a-z_]*: [0-9.]*' $L | tail -n 8 | tr '\n' ' '; echo
done
