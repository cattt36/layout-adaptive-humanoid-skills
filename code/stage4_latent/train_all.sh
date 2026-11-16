#!/bin/bash
# Train Z0..Z4 joint (3-action) latents with identical optimisation settings, for several seeds.
# env: LATENT_RUN (output name), ANNOT (annotation dir), GPU, SEEDS (default "42")
H=/root/gpufree-data/hnu_v2
OUT=$H/runs/${LATENT_RUN:-latent_v1}
mkdir -p $OUT
for S in ${SEEDS:-42}; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    CUDA_VISIBLE_DEVICES=${GPU:-1} bash $H/code/stage4_latent/train_latent.sh $P $OUT/seed$S --epochs 200 --seed $S \
      > $OUT/train_${P}_seed$S.log 2>&1 &
  done
  wait
done
echo all done
