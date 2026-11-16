#!/bin/bash
# Label-scarce Q_exec (1 % / 5 % of training envs): does level-allowed auxiliary supervision help when rollout labels
# are few (the real-robot regime)?  lambda in {0, 0.3, 1.0}, 3 seeds; lambda chosen on the calibration split.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
G=${1:-1}
cd $H/code
Q=$H/runs/quality_v4d
A=$H/runs/quality_aux_v4d
for FR in 0.01 0.05; do
  O=$H/runs/quality_aux_scarce_v4d/frac$FR
  for P in Z0 Z1 Z2 Z3 Z4; do
    for L in 0 0.3 1.0; do
      [ $P = Z0 ] && [ $L != 0 ] && continue
      for S in 0 1 2; do
        D=$O/$P/lam${L}_s$S
        [ -f $D/quality_eval.json ] && continue
        mkdir -p $D
        CUDA_VISIBLE_DEVICES=$G $PY stage5_quality/quality.py train --data $Q/$P/rollouts.npz --heuristic-dir $H/data/heuristic \
          --aux $A/aux_$P.npz --lambda-aux $L --seed $S --train-frac $FR --out $D > $D.log 2>&1
      done
    done
  done
  $PY tools/quality_aux_table.py $O > $O/table.md 2>&1
done
echo done > $H/runs/quality_aux_scarce_v4d/DONE
