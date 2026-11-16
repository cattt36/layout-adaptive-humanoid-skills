#!/bin/bash
# 4090, (user: adopt "preview all top-5 candidates in simulation" if it survives a robustness check).
# The E2E candidate sweep of tools/e2e_all.sh repeated with dynamics randomisation + latency (HNU_DR=1, identical to the
# closed-loop "randomised" environment) for the new students, RL seed 1, folds 1-4, L0-L3 -> results_candplan_xm*_dr1.
# Analysis (tools/preview_select.py, dr1 mode): preview in the ideal sweep (dr0) or in the randomised one (dr1, episodes
# 0..n-1), execution scored on dr1 episodes 8-15.  Starts after the external-baseline sweeps (tools/rwt_sweep.sh).
#   usage: preview_dr_sweep.sh   (two workers, GPU 0 / 1)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/preview_dr.log; }
for i in $(seq 1 400); do
  grep -q "variant t all folds done" $N/rwt_sweep.log 2>/dev/null && grep -q "variant s all folds done" $N/rwt_sweep.log && break
  sleep 30
done
log "start"
sweep() {  # fold level gpu
  local F=$1 P=$2 G=$3 FS=""; [ $F != 1 ] && FS=_f$F
  local SPL=$H/data/clip_split_cv/fold$F/split.json; [ $F = 1 ] && SPL=$H/data/clip_split_trial/split_v2.json
  local ST=$H/runs/students_v6g10xm${FS}_d2_s1/$P/student_2000.pt
  local K=$(python3 -c "import json;print(json.load(open('$H/data/z_cand_plan_f$F/pairs.json'))['K'])")
  local NETS=$(ls $H/runs/quality_v6g10n_tr_s1/$P/frac1.0_s*/quality_net.pt 2>/dev/null | paste -sd,)
  for k in $(seq -f %02g 0 $((K - 1))); do
    local O=$H/runs/results_candplan_xm${FS}_k${k}_s1_dr1 GK=$H/data/geom_cand_f$F/k$k
    [ -f $O/eval_student_$P.json ] && continue
    HNU_SPLIT_JSON=$SPL HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 HNU_EVAL_QEXEC=$NETS HNU_EVAL_CRIT=task \
      HNU_DR=1 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G nice $PY eval_student.py --student $ST --latent-dir $H/data/z_cand_plan_f$F/k$k/$P \
      --per-clip 16 --out $O --tag _$P >> $N/preview_dr_eval.log 2>&1
    [ -f $O/eval_student_$P.json ] || log "FAILED f$F $P k$k"
  done
  log "done f$F $P"
}
( for F in 1 2; do for P in Z0 Z1 Z2 Z3; do sweep $F $P 0; done; done ) &
( for F in 3 4; do for P in Z0 Z1 Z2 Z3; do sweep $F $P 1; done; done ) &
wait
log "all sweeps done"
