#!/bin/bash
# Progress: bridge students (v6r2), 0.10 m stake experts (g2_h10), robust 0.10 m stakes (rs10).
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== bridge students"; tail -n 3 $H/runs/night/v6r2.log
for s in 1 2 3; do for p in Z0 Z1 Z2 Z3 Z4; do
  a=$(tail -n 1 $H/runs/students_v6r2_s$s/$p/train_log.jsonl 2>/dev/null | grep -o '"it": [0-9]*' | cut -d' ' -f2)
  b=$(tail -n 1 $H/runs/students_v6r2_d2_s$s/$p/train_log.jsonl 2>/dev/null | grep -o '"it": [0-9]*' | cut -d' ' -f2)
  [ -n "$a$b" ] && echo "s$s $p dagger:${a:--} round2:${b:--}"
done; done
$PY tools/bridge_err.py $H/runs/results_v6r2_s1_dr0 2>/dev/null; $PY tools/bridge_err.py $H/runs/results_v6r2_s1_dr1 2>/dev/null
echo "== 0.10 m stake experts"
cat $H/runs/stakes_eval/g2_h10/summary.txt 2>/dev/null; cat $H/runs/stakes_eval/g2_h10/retention.txt 2>/dev/null
echo "== robust 0.10 m"
for f in ThreeStepPoleRun_rs10_mild FiveStepPoleRun_rs10_mild ThreeStepPoleRun_rs10_full FiveStepPoleRun_rs10_full; do
  L=$H/runs/logs/$f.log; [ -f $L ] || continue
  echo "$f: $(grep -a 'Learning iteration' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f4) | ep $(grep -a 'Mean episode length' $L | tail -n 1 | tr -s ' ' | cut -d' ' -f5)"
done
cat $H/runs/stakes_eval/robust_stakes_h10/summary.txt $H/runs/stakes_eval/robust_stakes_h10/retention.txt 2>/dev/null
