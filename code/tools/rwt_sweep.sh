#!/bin/bash
# 4090 server, external baseline "retrieve - warp - track" (tools/make_rwt.py) on the E2E candidate sweep.
# Tracker = the MoE expert bank the students are distilled from (runs/night/v6g10n_teachers.txt), evaluated exactly as
# the students' sweep (tools/e2e_all.sh): every (layout, library candidate) pair, 16 rollouts, nominal env, criterion
# v2 with the planned stance windows of data/geom_cand_f<k>.  Variants t / s (time only / time + space).
# The result of each table is linked as eval_student_Z0..Z3.json so that tools/expA_plan.py can treat the baseline like
# a student (one tracker; the level only changes the retrieval ranking / router of the E2E chain).  Seed label s1.
#   usage: rwt_sweep.sh <variant t|s> <gpu> [folds "1 2 3 4"]
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=$1 G=$2 FOLDS=${3:-"1 2 3 4"}
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rwt_sweep.log; }
CK=$(cat $N/v6g10n_teachers.txt)
for F in $FOLDS; do
  FS=""; [ $F != 1 ] && FS=_f$F
  SPL=$H/data/clip_split_cv/fold$F/split.json; [ $F = 1 ] && SPL=$H/data/clip_split_trial/split_v2.json
  [ -f $H/data/geom_rwt${V}_f$F/diag.json ] || $PY tools/make_rwt.py $F $V >> $N/rwt_sweep.log 2>&1 || { log "make_rwt f$F $V FAILED"; continue; }
  K=$(python3 -c "import json;print(json.load(open('$H/data/z_cand_plan_f$F/pairs.json'))['K'])")
  for k in $(seq -f %02g 0 $((K - 1))); do
    O=$H/runs/results_candplan_rwt$V${FS}_k${k}_s1_dr0 GK=$H/data/geom_rwt${V}_f$F/k$k
    if [ ! -f $O/eval_teacher_rwt.json ]; then
      HNU_SPLIT_JSON=$SPL HNU_BRIDGE_ROOT=$GK/bridge HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 \
        HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G nice $PY eval_student.py --teacher $CK --scene geom \
        --latent-dir $H/data/z_cand_plan_f$F/k$k/Z0 --per-clip 16 --out $O --tag _rwt >> $N/rwt_sweep_eval.log 2>&1
    fi
    if [ -f $O/eval_teacher_rwt.json ]; then
      for P in Z0 Z1 Z2 Z3; do ln -sf eval_teacher_rwt.json $O/eval_student_$P.json; done
    else
      log "FAILED f$F $V k$k"
    fi
  done
  log "done f$F variant $V"
done
log "variant $V all folds done"
