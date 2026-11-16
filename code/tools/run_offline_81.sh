#!/bin/bash
# Results 8.1 on the v2 latents: validation reports per action/profile, then the original Results scripts.
# usage: run_offline_81.sh <paper_root>
R=$1
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
F="$R/MOE Framework"
export PYTHONPATH="$F:$H/data/data_code"
for A in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    (cd "$F" && $PY validate_latents.py --latent-dir "$F/Latent/$A/$P" --profile $P --json-root $R/LLM_annotation \
       --datasets $A > "$F/Latent/$A/$P/validate.log" 2>&1) &
  done
done
wait
for A in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    (cd "$F" && $PY build_latent_index.py --latent-dir "$F/Latent/$A/$P" --profile $P --split train \
       --json-root $R/LLM_annotation --output "$F/Latent/$A/$P/latent_index_v3" > "$F/Latent/$A/$P/index.log" 2>&1) &
  done
done
wait
cd $R/Results/8.1_Offline_Representation/scripts
$PY compute_offline_latent_metrics.py --bootstrap-samples 1000 > ../latent_metrics.log 2>&1
$PY compute_offline_retrieval_metrics.py --bootstrap-samples 1000 > ../retrieval_metrics.log 2>&1
tail -n 5 ../latent_metrics.log ../retrieval_metrics.log
ls ../outputs
