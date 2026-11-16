#!/bin/bash
# 4090 server, (user request: E2E for ALL 48 new and 48 old students, i.e. folds 1-4 x L0-L3 x RL seeds 1-3,
# reported like 8.2-8.4, instruction and execution layer separately).  Replaces tools/e2e_folds.sh.
# One job = (fold, seed, student type, level): candidate sweep over every library candidate of the fold with the
# planner clock (tables of tools/make_z_cands_plan.py), 16 rollouts per (layout, candidate), nominal, criterion v2.
# Jobs start as soon as the student file is present (relayed from the L40 by tools/relay_students.ps1).  After the last
# job of a (fold, seed): tools/expA_plan.py <seed> <fold>; at the end tools/expA_pool.py.  The fold-1 seed-1 Q_exec nets
# are attached only to record falls.  Two workers (GPU 0 / 1).  Log runs/night/e2e_all.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/e2e_all.log; }
for F in 1 2 3 4; do [ -f $H/data/z_cand_plan_f$F/pairs.json ] || nice $PY tools/make_z_cands_plan.py $F >> $N/e2e_all.log 2>&1; done
student() { local FS=""; [ $1 != 1 ] && FS=_f$1; echo $H/runs/students_v6g10$3${FS}_d2_s$2/$4/student_2000.pt; }
sweep() {  # fold seed who level gpu
  local F=$1 s=$2 W=$3 P=$4 G=$5 FS=""; [ $F != 1 ] && FS=_f$F
  local SPL=$H/data/clip_split_cv/fold$F/split.json ST=$(student $F $s $W $P)
  [ $F = 1 ] && SPL=$H/data/clip_split_trial/split_v2.json
  local K=$(python3 -c "import json;print(json.load(open('$H/data/z_cand_plan_f$F/pairs.json'))['K'])")
  local NETS=$(ls $H/runs/quality_v6g10n_tr_s1/$P/frac1.0_s*/quality_net.pt 2>/dev/null | paste -sd,)
  for k in $(seq -f %02g 0 $((K - 1))); do
    local O=$H/runs/results_candplan_$W${FS}_k${k}_s${s}_dr0 GK=$H/data/geom_cand_f$F/k$k
    [ -f $O/eval_student_$P.json ] && continue
    HNU_SPLIT_JSON=$SPL HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 HNU_EVAL_QEXEC=$NETS HNU_EVAL_CRIT=task \
      HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST --latent-dir $H/data/z_cand_plan_f$F/k$k/$P \
      --per-clip 16 --out $O --tag _$P >> $N/e2e_all_eval.log 2>&1
    [ -f $O/eval_student_$P.json ] || log "FAILED f$F s$s $W $P k$k"
  done
}
fold_ready() { [ -f $H/runs/$([ $1 = 1 ] && echo latent_v5t || echo latent_cv_f$1)/seed42/Z3/language_routing_testv2.json ]; }
rank_ready() {
  local R=$H/runs/expA_rank$([ $1 != 1 ] && echo _f$1).json
  [ -f $R ] && return 0
  fold_ready $1 || return 1
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code nice $PY tools/expA_rank.py $1 >> $N/e2e_all.log 2>&1
  [ -f $R ]
}
JOBS=()
for s in 1 2 3; do for F in 1 2 3 4; do for W in xm n; do for P in Z0 Z3 Z1 Z2; do JOBS+=("$F $s $W $P"); done; done; done; done
Q=$N/e2e_all_claims
mkdir -p $Q
analyse() {  # fold seed: when all 8 sweeps of this (fold, seed) are complete
  local F=$1 s=$2 FS=""; [ $F != 1 ] && FS=_f$F
  for W in xm n; do for P in Z0 Z1 Z2 Z3; do [ -f $Q/done_${F}_${s}_${W}_$P ] || return; done; done
  mkdir $Q/analysed_${F}_$s 2> /dev/null || return
  PYTHONPATH=$H/data/data_code $PY tools/expA_plan.py $s $F > $H/runs/expA_plan${FS}_s$s.md 2>> $N/e2e_all.log
  log "analysis fold $F seed $s -> runs/expA_plan${FS}_s$s.md"
}
worker() {  # gpu
  local G=$1
  while true; do
    local left=0 took=0
    for i in "${!JOBS[@]}"; do
      set -- ${JOBS[$i]}
      [ -f $Q/done_${1}_${2}_${3}_$4 ] && continue
      left=1
      [ -f $(student $1 $2 $3 $4) ] || continue
      rank_ready $1 || continue
      mkdir $Q/claim_$i 2> /dev/null || continue
      sleep 30  # a relayed file settles
      log "start f$1 s$2 $3 $4 (gpu $G)"
      sweep $1 $2 $3 $4 $G
      touch $Q/done_${1}_${2}_${3}_$4
      log "done f$1 s$2 $3 $4"
      analyse $1 $2
      took=1
      break
    done
    [ $left = 0 ] && return
    [ $took = 0 ] && sleep 600
  done
}
log "start: ${#JOBS[@]} jobs"
worker 0 & sleep 10; worker 1 & wait
PYTHONPATH=$H/data/data_code $PY tools/expA_pool.py > $H/runs/expA_pool.md 2>> $N/e2e_all.log
log "all jobs complete -> runs/expA_pool.md"
