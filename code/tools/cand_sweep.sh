#!/bin/bash
# 4090 server, experiment A (full chain; runs/preregistration_expA_impl.md) -- candidate sweep.  Fold-1
# students of RL seed $SEED: every layout x every library clip of its action (tables of tools/make_z_cands.py),
# 16 rollouts, nominal, criterion task / v2, with the train-clip Q_exec nets attached (Q_exec^0 at the start state).
# Log: runs/night/cand_sweep.log
SEED=${SEED:-1}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/cand_sweep.log; }
S=$H/runs/students_v6g10n_d2_s$SEED
[ $(ls $S/Z{0,1,2,3}/student_2000.pt 2>/dev/null | wc -l) = 4 ] || { log "STOP: students missing (seed $SEED)"; exit 1; }
log "start seed $SEED (students $(md5sum $S/Z*/student_2000.pt | md5sum | cut -c1-8))"
[ -f $H/data/z_cand_f1/cands.json ] || $PY tools/make_z_cands.py 1 >> $N/cand_sweep.log 2>&1
K=$(python3 -c "import json;print(json.load(open('$H/data/z_cand_f1/cands.json'))['K'])")
lane() {  # gpu, jobs "k level"
  local G=$1; shift
  for j in "$@"; do
    set -- $j
    local O=$H/runs/results_v6g10n_cand_k$1_s${SEED}_dr0
    [ -f $O/eval_student_$2.json ] && [ -f $O/qexec_$2.npz ] && continue
    local NETS=$(ls $H/runs/quality_v6g10n_tr_s$SEED/$2/frac1.0_s*/quality_net.pt | paste -sd,)
    HNU_EVAL_QEXEC=$NETS HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
      --student $S/$2/student_2000.pt --latent-dir $H/data/z_cand_f1/k$1/$2 --per-clip 16 --out $O --tag _$2 \
      >> $N/cand_sweep_eval.log 2>&1
    log "k$1 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
  done
}
J=(); for k in $(seq -f %02g 0 $((K - 1))); do for P in Z0 Z3 Z1 Z2; do J+=("$k $P"); done; done  # L4 dropped
A=(); B=(); for i in "${!J[@]}"; do [ $((i % 2)) = 0 ] && A+=("${J[$i]}") || B+=("${J[$i]}"); done
lane 0 "${A[@]}" & lane 1 "${B[@]}" & wait
log "finished seed $SEED: $(ls $H/runs/results_v6g10n_cand_k*_s${SEED}_dr0/eval_student_Z*.json 2>/dev/null | wc -l) of $((K * 5))"
