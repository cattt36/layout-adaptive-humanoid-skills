#!/bin/bash
# Re-train v4 latents Z3/Z4 (seed 42) that hit CUDA OOM, then restart the v4 chain (resumes at the z-table step).
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for P in Z3 Z4; do
  rm -rf $H/runs/latent_v4/seed42/$P
  CUDA_VISIBLE_DEVICES=1 ANNOT=annot_v2 nice -n 15 bash stage4_latent/train_latent.sh $P $H/runs/latent_v4/seed42 --epochs 200 \
    --seed 42 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.10 \
    --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 \
    > $H/runs/latent_v4/train_${P}_seed42.log 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] fix_v4_z34 done" >> $H/runs/night/v4.log
nohup bash run_v4.sh > $H/runs/night/v4_launcher2.log 2>&1 < /dev/null &
