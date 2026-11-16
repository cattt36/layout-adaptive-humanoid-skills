#!/bin/bash
# v3 latents: identical loss weights for Z0..Z4 INCLUDING the skill-level transfer terms (lambda_skill_metric=0.10,
# lambda_skill_atoms=0.08; previously only Z4 had them).  Each level can then inject the supervision it owns into
# z_skill; Z0 has no metric/atom labels so the terms are inactive there.  CPU training (GPUs busy with students).
H=/root/gpufree-data/hnu_v2
OUT=$H/runs/latent_v3
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for S in ${SEEDS:-42 43 44}; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    CUDA_VISIBLE_DEVICES="${GPU:-}" ANNOT=annot_v2 bash $H/code/stage4_latent/train_latent.sh $P $OUT/seed$S --epochs 200 --seed $S \
      --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 > $OUT/train_${P}_seed$S.log 2>&1 &
  done
done
wait
echo all done
