#!/bin/bash
# Build a Data-HNU-like project root on the server so the original Results/ scripts run unchanged on the v2 latents.
# usage: setup_paper_root.sh <latent_run_dir e.g. runs/latent_v2/seed42> <annot_dir e.g. data/annot_v2>
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
LAT=$H/$1; ANN=$H/$2
R=$H/paper_root/$(basename $(dirname $LAT))_$(basename $LAT)
rm -rf $R; mkdir -p "$R/MOE Framework" $R/Results/8.1_Offline_Representation/scripts $R/Results/8.4_Language_Routing/scripts $R/Results/8.4_Language_Routing/inputs
# framework runtime = patched latent code + original MOE framework helpers + annotation schema modules
cp $H/code/stage4_latent/*.py "$R/MOE Framework/"
cp $H/data/moe_framework_py/*.py "$R/MOE Framework/" 2>/dev/null
cp -n $H/data/data_code/*.py $H/data/data_code/*.json "$R/MOE Framework/"
cp $H/code/stage4_latent/*.py "$R/MOE Framework/"   # patched versions win
cp -r $H/data/data_code $R/Data
cp -r $ANN/LLM_annotation $R/LLM_annotation
$PY $H/code/tools/make_latent_compat.py $LAT "$R/MOE Framework/Latent"
cp $H/code/results/compute_offline_*.py $R/Results/8.1_Offline_Representation/scripts/
cp $H/code/results/compute_router_ood_metrics.py $R/Results/8.4_Language_Routing/scripts/
cp $H/code/results/router_ood_benchmark_v1.json $R/Results/8.4_Language_Routing/inputs/
echo $R
