#!/bin/bash
# Results 8.1 for an L3-kin paper root (tools/run_l3kin.sh): identical to tools/run_offline_81.sh except that the
# Z3 validators / index builders run with HNU_HUMAN_DYNAMICS=0 (the L3-kin checkpoint was trained without human
# dynamics, so its semantic numeric inputs must be built the same way).  Z0 / Z1 / Z2 / Z4 are the standard latents.
# usage: run_offline_81_kin.sh <paper_root>
R=$1
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
F="$R/MOE Framework"
export PYTHONPATH="$F:$H/data/data_code"
hd() { [ $1 = Z3 ] && echo ${Z3_DYNAMICS:-0} || echo 1; }  # L3-kin / L3-rom: 0 (default); L3-noQ: Z3_DYNAMICS=1
ht() { [ $1 = Z3 ] && echo ${Z3_TIMING:-1} || echo 1; }  # L3-rom (run_l3rom.sh): Z3_TIMING=0
for A in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    (cd "$F" && HNU_HUMAN_DYNAMICS=$(hd $P) HNU_HUMAN_TIMING=$(ht $P) $PY validate_latents.py --latent-dir "$F/Latent/$A/$P" --profile $P \
       --json-root $R/LLM_annotation --datasets $A > "$F/Latent/$A/$P/validate.log" 2>&1) &
  done
done
wait
for A in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    (cd "$F" && HNU_HUMAN_DYNAMICS=$(hd $P) HNU_HUMAN_TIMING=$(ht $P) $PY build_latent_index.py --latent-dir "$F/Latent/$A/$P" --profile $P --split train \
       --json-root $R/LLM_annotation --output "$F/Latent/$A/$P/latent_index_v3" > "$F/Latent/$A/$P/index.log" 2>&1) &
  done
done
wait
cd $R/Results/8.1_Offline_Representation/scripts
$PY compute_offline_latent_metrics.py --bootstrap-samples 1000 > ../latent_metrics.log 2>&1
$PY compute_offline_retrieval_metrics.py --bootstrap-samples 1000 > ../retrieval_metrics.log 2>&1
tail -n 5 ../latent_metrics.log ../retrieval_metrics.log
ls ../outputs
