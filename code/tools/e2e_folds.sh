#!/bin/bash
# 4090 server, (user request: E2E on all 4 folds, without affecting the L40 main study).  Per fold k = 2..4,
# as soon as its inputs exist: routers (tools/e2e_router_folds.sh) -> candidate tables with the planner
# (make_z_cands_plan.py k) -> retrieval ranking (expA_rank.py k) -> wait for the seed-1 L0 / L3 students, new (v6g10xm)
# and old (v6g10n), relayed from the L40 by tools/relay_students.ps1 -> candidate sweep (cand_plan_sweep.sh, FOLD=k)
# -> analysis runs/expA_plan_f<k>_s1.md.  Then the pooled 4-fold table (tools/expA_pool.py).  Log runs/night/e2e_folds.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export PYTHONPATH=$H/data/data_code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/e2e_folds.log; }
log "start"
for F in 2 3 4; do
  [ -d $H/data/z_cand_plan_f$F ] || nice $PY tools/make_z_cands_plan.py $F >> $N/e2e_folds.log 2>&1
done
left="2 3 4"
for i in $(seq 1 200); do
  rest=""
  for F in $left; do
    R=$H/runs/latent_cv_f$F/seed42
    if [ ! -f $R/Z0/language_routing_testv2.json ] || [ ! -f $R/Z3/language_routing_testv2.json ]; then rest="$rest $F"; continue; fi
    [ -f $H/runs/expA_rank_f$F.json ] || { CUDA_VISIBLE_DEVICES= nice $PY tools/expA_rank.py $F >> $N/e2e_folds.log 2>&1; log "fold $F ranking done"; }
    ok=1
    for W in xm n; do for P in Z0 Z3; do [ -f $H/runs/students_v6g10${W}_f${F}_d2_s1/$P/student_2000.pt ] || ok=0; done; done
    if [ $ok = 0 ]; then rest="$rest $F"; continue; fi
    sleep 120  # let a relayed file settle
    log "fold $F sweep start"
    FOLD=$F SEED=1 bash tools/cand_plan_sweep.sh
    $PY tools/expA_plan.py 1 $F > $H/runs/expA_plan_f${F}_s1.md 2>> $N/e2e_folds.log
    log "fold $F analysis -> runs/expA_plan_f${F}_s1.md"
  done
  left=$rest
  [ -z "$(echo $left)" ] && break
  sleep 300
done
$PY tools/expA_pool.py 1 > $H/runs/expA_pool_s1.md 2>> $N/e2e_folds.log
log "all folds: runs/expA_pool_s1.md"
