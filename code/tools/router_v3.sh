#!/bin/bash
# 4090 server, CPU only, (user request: improve the instruction layer, methods 1 + 2, same for every level).
# 1  tools/make_instructions_aug_v3.py  -> data_lang/instructions_aug_v3.json (implicit + ASR-noise training commands,
#    out-of-scope / ambiguous bank; filtered against val2 / benchmark / sealed test)
# 2  router_finetune2.py --extra ... --tag ft3 on every fold encoder (the frozen procedure otherwise unchanged)
# 3  stage4_latent/router_gate3.py: 2 x 2 ablation (ft2/ft3 encoder x old/new gate), writes language_routing_testv3.json
# Log: runs/night/router_v3.log; table: runs/router_v3.md
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
export PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/router_v3.log; }
log "start"
$PY tools/make_instructions_aug_v3.py >> $N/router_v3.log 2>&1 || { log "aug FAILED"; exit 1; }
for R in latent_v5t latent_cv_f2 latent_cv_f3 latent_cv_f4; do
  ( nice $PY stage4_latent/router_finetune2.py data_lang/instructions_v1.json data_lang/instructions_val_v2.json \
      results/router_ood_benchmark_v1.json $H/runs/$R --extra data_lang/instructions_aug_v3.json --tag ft3 \
      > $N/router_v3_ft3_$R.log 2>&1 && log "ft3 $R ok" || log "ft3 $R FAILED" ) &
done
wait
nice $PY stage4_latent/router_gate3.py $H/runs/latent_v5t $H/runs/latent_cv_f2 $H/runs/latent_cv_f3 $H/runs/latent_cv_f4 \
  > $H/runs/router_v3.md 2>> $N/router_v3.log && log "gate3 ok" || log "gate3 FAILED"
log "all steps done"
