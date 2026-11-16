#!/bin/bash
# L40, (user request: start the old-student completion now on the idle GPUs instead of waiting for the end of
# run_v6g10xm_l40.sh, whose remaining jobs are the fold-1 L3 deployment seeds 4 / 5).  Identical jobs, recipe and
# evaluation as tools/old_complete_l40.sh (that waiting instance is stopped first):
#   1. missing old (v6g10n) seed-3 students, folds 1-4 x L0-L3 (one() of run_v6g10n_l40b.sh)
#   2. condition-B evaluation of every old student (plan4; task x DR 0/1 x push / no push) -> results_v6g10nB*
# Workers start at once on the GPUs given in NOW_GPUS (default "1 2"); workers for LATER_GPUS (default "0 3 4") join
# as soon as run_v6g10xm_l40.sh has ended.  Jobs are claimed through the same directory queue, so nothing runs twice.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
NOW_GPUS=${NOW_GPUS:-"1 2"}
LATER_GPUS=${LATER_GPUS:-"0 3 4"}
source <(sed -n '17,77p' run_v6g10n_l40b.sh | sed '/^# teacher bank on this hardware/,/^log "teacher bank evaluated"$/d')
log "old_complete (early start on GPUs $NOW_GPUS; $LATER_GPUS after the new study): missing seed-3 students, then condition B"
evalB() {  # fold P gpu seed
  local F=$1 P=$2 G=$3 s=$4 VS=v6g10n
  [ $F != 1 ] && VS=${VS}_f$F
  local ST=$H/runs/students_${VS}_d2_s$s/$P/student_2000.pt RB=$H/runs/results_v6g10nB$([ $F != 1 ] && echo _f$F) GP=$H/data/geom_plan4_f$F
  [ -f $ST ] || { log "old_complete: missing $ST"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    [ -f ${RB}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
    HNU_GEOM_W1=$GP/ref_stakes_s93_h10 HNU_GEOM_W2=$GP/ref_stakes_s83_h10 HNU_SPLIT_JSON=$H/data/clip_split_cv/fold$F/split.json \
      HNU_EVAL_CRIT=task HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST \
      --latent-dir $H/data/z_lib_f$F/plan4/$P --per-clip 32 $X --out ${RB}_s${s}_dr$DRV --tag _$P >> $N/old_complete_eval.log 2>&1
  done; done
}
JOBS=()
for F in 1 2 3 4; do for P in Z0 Z3 Z1 Z2; do JOBS+=("T $F $P 3"); done; done
for s in 1 2; do for F in 1 2 3 4; do for P in Z0 Z3 Z1 Z2; do JOBS+=("B $F $P $s"); done; done; done  # seed 3: after its T job
Q=$N/old_complete_claims
rm -rf $Q; mkdir -p $Q
worker() {  # gpu (G)
  for i in "${!JOBS[@]}"; do
    mkdir $Q/$i 2> /dev/null || continue
    set -- ${JOBS[$i]}
    if [ $1 = T ]; then one $2 $3 $G $4; fi
    evalB $2 $3 $G $4
  done
}
for G in $NOW_GPUS; do G=$G worker & sleep 20; done
(
  while pgrep -f "bash run_v6g10xm_l40.sh" > /dev/null; do sleep 300; done
  log "old_complete: new study ended, adding workers on GPUs $LATER_GPUS"
  for G in $LATER_GPUS; do G=$G worker & sleep 20; done
  wait
) &
wait
log "old_complete: all old students trained and evaluated under condition B"
