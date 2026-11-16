#!/bin/bash
# 4-fold CV students, folds 2-4 (pre-registered , runs/preregistration_cv.md; fold 1 = v6g10n seeds 1-3).
# L0-L4 x RL seeds 1-3 per fold (45 students), recipe IDENTICAL to run_v6g10n.sh (same teachers, augmentation, DAgger
# 4000 + 2000, evaluation protocol); only the fold's data differ: training trials + tempo copies (fold<k>/
# clips_train_aug.txt, aug_ts_f<k>), z tables of the fold's own encoders (z_tables_cv_f<k>, seed-42 latents) and the
# fold's split labels for the evaluation.  Baselines (REF / No-latent) only in fold 1.
# Starts when v6g10n has finished (one student per 24 GB GPU) and fold k's data are ready (tools/run_cv_prep.sh).
# Log: runs/night/cv_students.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10n
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold -> training augmentation (as run_v6g10n.sh, fold-specific geometry dirs)
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_ts_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/cv_students.log; }
ready() { grep -q "fold $1 training set: .*problems: 0" $N/cv_prep.log 2>/dev/null; }
log "waiting for v6g10n to finish and for the fold data (run_cv_prep.sh)"
until grep -q "students done" $N/$V.log 2>/dev/null; do sleep 300; done
CK=$(cat $N/${V}_teachers.txt)
log "start: teachers $(echo $CK | tr '\n' ' ')"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # fold P gpu seed
  local F=$1 P=$2 G=$3 s=$4
  local S=$H/runs/students_${V}_f${F}_s$s O=$H/runs/students_${V}_f${F}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P \
    LT=$H/data/z_tables_cv_f${F}_aug/$P ALLOW=$H/data/clip_split_cv/fold$F/clips_train_aug.txt AUG=$(aug $F)
  until ready $F; do sleep 300; done
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_f${F}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_f${F}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${V}_f${F}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F $P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_f${F}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_f${F}_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${V}_f${F}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F $P s$s round 2 failed"; return; }
  for CR in task track; do
    local R=$H/runs/results_${V}_f$F; [ $CR = track ] && R=$H/runs/results_${V}trk_f$F
    for DRV in 0 1; do for X in "" "--push"; do
      [ -f ${R}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
      HNU_SPLIT_JSON=$H/data/clip_split_cv/fold$F/split.json HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 \
        CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
        --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${V}_f${F}_F_s${s}_$P.log 2>&1
    done; done
  done
  log "fold $F $P s$s done"
}
# order: seed 1 of every fold first (L0 / L4 first within a fold), so an interrupted run stays balanced across folds
JOBS=(); for s in 1 2 3; do for F in 2 3 4; do for P in Z0 Z4 Z1 Z2 Z3; do JOBS+=("$F $P $s"); done; done; done
Q=$N/cv_students_queue; echo 0 > $Q.next; : > $Q.lock
worker() {
  while true; do
    while [ "$(free_mb $1)" -lt 17000 ]; do sleep 60; done
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    set -- $1 ${JOBS[$i]}
    one $2 $3 $1 $4
  done
}
worker 0 & worker 1 & wait
for j in "${JOBS[@]}"; do set -- $j; g=1; [ "$(free_mb 0)" -ge 17000 ] && g=0; one $1 $2 $g $3; done  # retry failures
n=$(ls $H/runs/students_${V}_f*_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 45"
[ "$n" = 45 ] && log "finished"
