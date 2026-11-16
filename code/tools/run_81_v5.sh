#!/bin/bash
# Results 8.1 (offline representation) for the v5 latents (the latents of the deployed z_skill tables), 3 seeds, CPU only.
H=/root/gpufree-data/hnu_v2
cd $H/code
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for s in 42 43 44; do
  R=$(bash tools/setup_paper_root.sh runs/latent_v5/seed$s data/annot_v2 | tail -n 1)
  bash tools/run_offline_81.sh $R > $H/runs/night/offline81_v5_seed$s.log 2>&1
  echo "[$(date +%H:%M:%S)] seed $s done: $(ls $R/Results/8.1_Offline_Representation/outputs 2>/dev/null | wc -l) output files" >> $H/runs/night/offline81_v5.log
done
/root/gpufree-data/venvs/mjlab/bin/python tools/aggregate_81.py $H/paper_root > $H/runs/night/offline81_summary_all.md 2>&1
echo "[$(date +%H:%M:%S)] finished" >> $H/runs/night/offline81_v5.log