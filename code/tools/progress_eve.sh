#!/bin/bash
# Progress: 0.10 m stakes (g2), robust stakes (rs), robust bridge/flat (v6r2), v6g.
H=/root/gpufree-data/hnu_v2
date '+%m-%d %H:%M'
for f in ThreeStepPoleRun_g2_h08 FiveStepPoleRun_g2_h08 ThreeStepPoleRun_g2_h10 FiveStepPoleRun_g2_h10 ThreeStepPoleRun_rs_mild FiveStepPoleRun_rs_mild \
         ThreeStepPoleRun_rs_full FiveStepPoleRun_rs_full CrouchedTraverse_r2_full_bridge CrouchedTraverse_r2_more_bridge ThreeStepPoleRun_r2_more_flat; do
  L=$H/runs/logs/$f.log; [ -f $L ] || continue
  echo "$f: $(grep -a 'Learning iteration' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f4) | ep $(grep -a 'Mean episode length' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f5) | $(grep -a -o 'stake_missed: [0-9.]*' $L | tail -n 1)"
done
echo "== v6r2 log"; tail -n 4 $H/runs/night/v6r2.log; cat $H/runs/night/v6r2_retention.log 2>/dev/null
echo "== v6g log"; cat $H/runs/night/v6g.log
cat $H/runs/stakes_eval/g2_h10/summary.txt $H/runs/stakes_eval/robust_stakes/summary.txt 2>/dev/null
