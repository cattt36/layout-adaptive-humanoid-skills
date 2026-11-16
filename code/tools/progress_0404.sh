#!/bin/bash
# Progress snapshot: gated stake runs, robust teachers (v6r2), v5 randomised baseline.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== stakes (gate)"
for r in gate_flat gate_h06; do for T in ThreeStepPoleRun FiveStepPoleRun; do
  L=$H/runs/logs/${T}_$r.log
  echo "$T $r: $(grep -a 'Learning iteration' $L | tail -n 1 | tr -s ' ') | ep $(grep -a 'Mean episode length' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f5) | stake_missed $(grep -a -o 'stake_missed: [0-9.]*' $L | tail -n 1)"
done; done
cat $H/runs/stakes_eval/gate_summary.txt 2>/dev/null
cat $H/runs/stakes_eval/gate_retention.txt 2>/dev/null
echo "== robust teachers (v6r2)"
cat $H/runs/night/v6r2.log | tail -n 4
for f in CrouchedTraverse_r2_mild_bridge ThreeStepPoleRun_r2_mild_flat FiveStepPoleRun_r2_mild_flat CrouchedTraverse_r2_full_bridge; do
  L=$H/runs/logs/$f.log; [ -f $L ] || continue
  echo "$f: $(grep -a 'Learning iteration' $L | tail -n 1 | tr -s ' ') | ep $(grep -a 'Mean episode length' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f5)"
done
echo "== v5 in randomised env"
cat $H/runs/night/v5_dr1_summary.txt 2>/dev/null
