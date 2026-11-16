#!/bin/bash
# z_quality auxiliary supervision (level-allowed robot / human metrics), on the existing v4d Q_exec rollouts.
# lambda_aux in {0, 0.1, 0.3, 1.0} x 3 training seeds per level; lambda chosen per level on the CALIBRATION split
# (cal AUROC, mean over seeds); the test split is only read for the chosen setting (tools/quality_aux_table.py).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
G=${1:-1}
cd $H/code
Q=$H/runs/quality_v4d
O=$H/runs/quality_aux_v4d
mkdir -p $O
for P in Z0 Z1 Z2 Z3 Z4; do
  $PY stage5_quality/make_quality_aux.py --annot $H/data/annot_v2/LLM_annotation --ref $H/data/ref_v1 --level $P \
    --forceplate $H/data/forceplate --out $O/aux_$P.npz >> $O/aux.log 2>&1
done
for P in Z0 Z1 Z2 Z3 Z4; do
  for L in 0 0.1 0.3 1.0; do
    [ $P = Z0 ] && [ $L != 0 ] && continue
    for S in 0 1 2; do
      D=$O/$P/lam${L}_s$S
      [ -f $D/quality_eval.json ] && continue
      mkdir -p $D
      CUDA_VISIBLE_DEVICES=$G $PY stage5_quality/quality.py train --data $Q/$P/rollouts.npz --heuristic-dir $H/data/heuristic \
        --aux $O/aux_$P.npz --lambda-aux $L --seed $S --out $D > $D.log 2>&1
    done
  done
done
$PY tools/quality_aux_table.py $O > $O/table.md 2>&1
echo done > $O/DONE
