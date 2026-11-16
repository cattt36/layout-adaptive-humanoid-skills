#!/bin/bash
# 4090 server, full chain with the geometric planner -- candidate sweep (tools/make_z_cands_plan.py).
# Fold-1 RL seed $SEED students, new (v6g10xm, mix recipe) and old (v6g10n), L0 / L3: every layout x every library
# clip of its action with the planner clock of that pair, 16 rollouts, nominal, criterion task / v2.  The old Q_exec
# nets are attached only to record falls per rollout (their Q values are not used for the new students).
# Log: runs/night/cand_plan.log
# FOLD=2..4 (E2E on all folds): the fold's students (students_v6g10{xm,n}_f<k>_d2_s<seed>), candidate tables
# data/z_cand_plan_f<k>, split clip_split_cv/fold<k>/split.json; results_candplan_<who>_f<k>_k*.  The fold-1 Q_exec nets
# are attached only to record falls.
SEED=${SEED:-1}
FOLD=${FOLD:-1}
FS=""; [ $FOLD != 1 ] && FS=_f$FOLD
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 \
  HNU_POST_SHAPE=mesh12 HNU_SPLIT_JSON=$([ $FOLD = 1 ] && echo $H/data/clip_split_trial/split_v2.json || echo $H/data/clip_split_cv/fold$FOLD/split.json)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/cand_plan$FS.log; }
[ -f $H/data/z_cand_plan_f$FOLD/pairs.json ] || $PY tools/make_z_cands_plan.py $FOLD >> $N/cand_plan.log 2>&1
K=$(python3 -c "import json;print(json.load(open('$H/data/z_cand_plan_f$FOLD/pairs.json'))['K'])")
log "start seed $SEED, K=$K"
lane() {  # gpu, jobs "k who level"
  local G=$1; shift
  for j in "$@"; do
    set -- $j
    local O=$H/runs/results_candplan_$2${FS}_k$1_s${SEED}_dr0 ST=$H/runs/students_v6g10$2${FS}_d2_s$SEED/$3/student_2000.pt GK=$H/data/geom_cand_f$FOLD/k$1
    [ -f $O/eval_student_$3.json ] && continue
    [ -f $ST ] || { log "missing $ST"; continue; }
    local NETS=$(ls $H/runs/quality_v6g10n_tr_s$SEED/$3/frac1.0_s*/quality_net.pt 2>/dev/null | paste -sd,)
    HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 HNU_EVAL_QEXEC=$NETS HNU_EVAL_CRIT=task HNU_DR=0 \
      HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST --latent-dir $H/data/z_cand_plan_f$FOLD/k$1/$3 \
      --per-clip 16 --out $O --tag _$3 >> $N/cand_plan${FS}_eval.log 2>&1
    log "k$1 $2 $3: $([ -f $O/eval_student_$3.json ] && echo ok || echo FAILED)"
  done
}
J=(); for k in $(seq -f %02g 0 $((K - 1))); do for W in xm n; do for P in Z3 Z0; do J+=("$k $W $P"); done; done; done
A=(); B=(); for i in "${!J[@]}"; do [ $((i % 2)) = 0 ] && A+=("${J[$i]}") || B+=("${J[$i]}"); done
lane 0 "${A[@]}" & lane 1 "${B[@]}" & wait
log "sweep complete: $(ls $H/runs/results_candplan_*${FS}_k*_s${SEED}_dr0/eval_student_Z*.json 2>/dev/null | wc -l) of $((K * 4))"
