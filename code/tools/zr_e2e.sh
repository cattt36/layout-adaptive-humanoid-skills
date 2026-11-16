#!/bin/bash
# 4090, E2E candidate sweep for the fair baseline ZR (explicit-reference students, RL seed 1, folds 1-4;
# relayed from the L40).  ZR tables of the original / tempo clips (tools/make_zr.py --no-xz, identical normalisation),
# planner candidate tables for level ZR (make_z_cands_plan.py, geometry untouched), the same sweep as tools/e2e_all.sh
# (16 rollouts per layout x candidate, nominal, criterion v2) -> results_candplan_zr[_f<k>]_k*_s1_dr0 with the result
# linked as eval_student_Z0..Z3.json (one student; the level only changes routing / retrieval), then E2E pools with
# HNU_WHO_EXTRA=zr (frozen router and router v4) and the preview selection for ZR.  Log runs/night/zr_e2e.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
D=$H/data
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/zr_e2e.log; }
SEEDS=${SEEDS:-1}  # SEEDS="2 3" for the added ZR seeds (pools then use every seed present)
log "start (seeds $SEEDS)"
for S in $SEEDS; do
for F in 1 2 3 4; do
  FS=""; [ $F != 1 ] && FS=_f$F
  ST=$H/runs/students_v6g10xm${FS}_d2_s$S/ZR/student_2000.pt
  for i in $(seq 1 240); do [ -f $ST ] && [ -f $ST.ok ] && break; sleep 30; done
  [ -f $ST.ok ] || { log "fold $F student missing"; continue; }
  [ -d $D/z_tables_cv_f$F/ZR ] || $PY tools/make_zr.py $F --no-xz >> $N/zr_e2e.log 2>&1
  [ -d $D/z_cand_plan_f$F/k00/ZR ] || HNU_LEVELS=ZR HNU_SKIP_GEOM=1 $PY tools/make_z_cands_plan.py $F >> $N/zr_e2e.log 2>&1
  SPL=$D/clip_split_cv/fold$F/split.json; [ $F = 1 ] && SPL=$D/clip_split_trial/split_v2.json
  K=$(python3 -c "import json;print(json.load(open('$D/z_cand_plan_f$F/pairs.json'))['K'])")
  # per-rollout outcomes (preview analysis) via HNU_EVAL_EPISODES: the z-student Q_exec nets do not fit ZR's observation
  for k in $(seq -f %02g 0 $((K - 1))); do
    O=$H/runs/results_candplan_zr${FS}_k${k}_s${S}_dr0 GK=$D/geom_cand_f$F/k$k
    if [ ! -f $O/eval_student_ZR.json ]; then
      HNU_SPLIT_JSON=$SPL HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 HNU_EVAL_EPISODES=1 HNU_EVAL_CRIT=task \
        HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=0 nice $PY eval_student.py --student $ST --latent-dir $D/z_cand_plan_f$F/k$k/ZR \
        --per-clip 16 --out $O --tag _ZR >> $N/zr_e2e_eval.log 2>&1
    fi
    if [ -f $O/eval_student_ZR.json ]; then
      for P in Z0 Z1 Z2 Z3; do ln -sf eval_student_ZR.json $O/eval_student_$P.json; [ -f $O/qexec_ZR.npz ] && ln -sf qexec_ZR.npz $O/qexec_$P.npz; done
    else
      log "FAILED f$F k$k"
    fi
  done
  log "fold $F seed $S swept"
done
done
export PYTHONPATH=$D/data_code CUDA_VISIBLE_DEVICES= HNU_WHO_EXTRA=zr
for RT in "" v4; do
  for S in $SEEDS; do for F in 1 2 3 4; do
    FS=""; [ $F != 1 ] && FS=_f$F
    [ -f $H/runs/expA_plan${FS}_s${S}_rows.json ] && HNU_RTAG=$RT nice $PY tools/expA_plan.py $S $F > /dev/null 2>> $N/zr_e2e.log &
  done; done
  wait
  HNU_RTAG=$RT nice $PY tools/expA_pool.py > $H/runs/expA_pool${RT:+_$RT}_zr.md 2>> $N/zr_e2e.log
done
HNU_RTAG=v4 nice $PY tools/e2e_refusal_pool.py > $H/runs/e2e_refusal_v4_zr.md 2>> $N/zr_e2e.log
for S in $SEEDS; do $PY tools/preview_select.py zr $S > $H/runs/preview_zr_s$S.md 2>> $N/zr_e2e.log; done
log "all steps done (seeds $SEEDS) -> runs/expA_pool_zr.md, expA_pool_v4_zr.md, e2e_refusal_v4_zr.md, preview_zr_s*.md"
