#!/bin/bash
# 4090 server, CPU only, E2E with the router v3 variants (no new simulation: the candidate sweep is reused,
# only which commands are accepted and how the library is ranked changes).
#   1 stage4_latent/router_gate3.py (all variants, writes language_routing_test<code>.json)
#   2 per code: tools/expA_rank.py <fold> -> expA_plan.py <seed> <fold> (every (fold, seed) whose frozen rows exist)
#     -> expA_pool.py -> runs/expA_pool_<code>.md
# Log: runs/night/e2e_v3.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
export PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/e2e_v3.log; }
CODES=${CODES:-"v3 v3n90 v3o v3o90 v2o90"}
log "start ($CODES)"
nice $PY stage4_latent/router_gate3.py $H/runs/latent_v5t $H/runs/latent_cv_f2 $H/runs/latent_cv_f3 $H/runs/latent_cv_f4 \
  > $H/runs/router_v3.md 2>> $N/e2e_v3.log || { log "gate3 FAILED"; exit 1; }
log "gate3 ok"
for C in $CODES; do
  for F in 1 2 3 4; do
    ( HNU_RTAG=$C nice $PY tools/expA_rank.py $F > /dev/null 2>> $N/e2e_v3.log || log "rank $C f$F FAILED" ) &
  done
  wait
  for F in 1 2 3 4; do
    FS=$([ $F = 1 ] && echo "" || echo "_f$F")
    for S in 1 2 3; do
      [ -f $H/runs/expA_plan${FS}_s${S}_rows.json ] || continue
      ( HNU_RTAG=$C nice $PY tools/expA_plan.py $S $F > $H/runs/expA_plan${FS}_s${S}_$C.md 2>> $N/e2e_v3.log || log "plan $C f$F s$S FAILED" ) &
    done
  done
  wait
  HNU_RTAG=$C nice $PY tools/expA_pool.py > $H/runs/expA_pool_$C.md 2>> $N/e2e_v3.log && log "pool $C ok" || log "pool $C FAILED"
done
log "all steps done"
