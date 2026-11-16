#!/bin/bash
# v4b latents (= v4 + routing loss lambda_query_task 0.2, fixed a priori) (same settings for Z0..Z4):
#   skill-level transfer (robot metrics/atoms) 0.10/0.08, human-only atom head on z_skill and z_sem 0.10/0.10,
#   frozen multilingual sentence encoder as text front-end (384-d), query augmentation p=0.5 with the
#   descriptions each profile owns (robot atoms from L2, human captions/hints/atoms at L4).
# env: SEEDS (default "42 43 44"), GPU (default 1), EPOCHS (200), OUT (runs/latent_v4)
H=/root/gpufree-data/hnu_v2
OUT=${OUT:-$H/runs/latent_v4b}
mkdir -p $OUT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for S in ${SEEDS:-42 43 44}; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    CUDA_VISIBLE_DEVICES=${GPU:-1} ANNOT=annot_v2 nice -n 15 bash $H/code/stage4_latent/train_latent.sh $P $OUT/seed$S --epochs ${EPOCHS:-200} \
      --seed $S --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.10 \
      --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 \
      > $OUT/train_${P}_seed$S.log 2>&1 &
  done
  [ "${PARALLEL_SEEDS:-0}" = 1 ] || wait
done
wait
echo all done
