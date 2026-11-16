#!/bin/bash
# Results 8.1 on the trial-split latents v5t (metric scripts now accept the explicit held-out split).
H=/root/gpufree-data/hnu_v2
cd $H/code
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for s in 42 43 44; do
  R=$(bash tools/setup_paper_root.sh runs/latent_v5t/seed$s data/annot_v2 | tail -n 1)
  bash tools/run_offline_81.sh $R > $H/runs/night/offline81_v5t_seed$s.log 2>&1
  grep -h -e Error -e Traceback $R/Results/8.1_Offline_Representation/*.log | head -n 3
done
/root/gpufree-data/venvs/mjlab/bin/python tools/aggregate_81.py $H/paper_root > $H/runs/night/offline81_summary_all.md 2>&1
grep -A 24 "latent_v5t (3 seeds)" $H/runs/night/offline81_summary_all.md
grep -A 12 "### latent_v5 (3 seeds)" $H/runs/night/offline81_summary_all.md | head -n 14
