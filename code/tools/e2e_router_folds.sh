#!/bin/bash
# 4090 server, CPU only, (user request: E2E on all 4 folds, must not affect the L40 main study).
# The frozen router procedure of run_latent_t.sh (router_finetune2 -> router_abstain -> final_router_eval on the sealed
# test set), applied unchanged to the fold encoders runs/latent_cv_f<k> (k = 2..4).  Fold 1 = runs/latent_v5t (done).
# Log: runs/night/e2e_router_f<k>.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
for F in 2 3 4; do
  R=$H/runs/latent_cv_f$F
  [ -f $R/seed42/Z3/language_routing_testv2.json ] && continue
  echo "[$(date '+%m-%d %H:%M:%S')] fold $F start" >> $N/e2e_router_f$F.log
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code nice $PY stage4_latent/router_finetune2.py data_lang/instructions_v1.json \
    data_lang/instructions_val_v2.json results/router_ood_benchmark_v1.json $R >> $N/e2e_router_f$F.log 2>&1
  HNU_ABSTAIN_SPACE=query HNU_INSTRUCTIONS=data_lang/instructions_v1.json CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code \
    nice $PY stage4_latent/router_abstain.py data_lang/instructions_val_v2.json results/router_ood_benchmark_v1.json $R ft2 \
    >> $N/e2e_router_f$F.log 2>&1
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code nice $PY stage4_latent/final_router_eval.py data_lang/router_test_v2_sealed.json \
    data_lang/instructions_v1.json data_lang/instructions_val_v2.json $R > $N/router_test_v2_cv_f$F.md 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] fold $F router done: $(ls $R/seed*/Z*/language_routing_testv2.json 2>/dev/null | wc -l) dirs" >> $N/e2e_router_f$F.log
done
