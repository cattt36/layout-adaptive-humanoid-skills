#!/bin/bash
# v6g10xm main study on the L40 server (user decision): students with cross-trial z augmentation, recipe
# "mix" (tools/make_xz.py <fold> mix: per training clip one linear cross-z copy, one stance-aligned cross-z copy and one
# own-z duplicate -> 50 % mismatched z).  Everything else identical to run_v6g10n_l40b.sh (teachers, augmentation,
# DAgger 4000 + 2000, per-fold data, seeds).  Jobs: folds 1-4 x L0-L3 x seeds 1-3 (fold-1 seed-1 L0 / L3 exist from
# the pilot, tools/xz_r2.sh, and are skipped), then fold-1 L3 seeds 4-5 (deployment seed selection).
# Evaluation per student (split of its fold, 32 rollouts per clip):
#   condition A (pre-registered; own z, human clock): task / track criterion x DR 0 / 1 x push / no push
#                -> results_v6g10xm[_f<k>]_s<s>_dr*, results_v6g10xmtrk[_f<k>]_s<s>_dr*
#   condition B (deployable; nearest library z of the fold's training clips + geometric planner, plan4):
#                task criterion x DR 0 / 1 x push / no push -> results_v6g10xmB[_f<k>]_s<s>_dr*
# Log runs/night/v6g10xm.log (the word that ends the watchdog is written only once, at the very end).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10xm
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p $N
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_xzmix_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_xzmix_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_xzmix_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/$V.log; }
CK=$(cat $N/v6g10n_teachers.txt)
log "start (L40, $NGPU GPUs): teachers $CK"
# per-fold data: mix training set, library tables, planner (plan4)
for F in 1 2 3 4; do
  [ -s $H/data/clip_split_cv/fold$F/clips_train_xzmix.txt ] || $PY tools/make_xz.py $F mix >> $N/$V.log 2>&1
  [ -d $H/data/z_lib_f$F/geo/Z3 ] || $PY tools/make_z_library.py $F >> $N/$V.log 2>&1
  [ -d $H/data/z_lib_f$F/plan4/Z3 ] || $PY tools/make_z_plan.py $F plan4 >> $N/$V.log 2>&1
  [ -s $H/data/clip_split_cv/fold$F/clips_train_xzmix.txt ] && [ -d $H/data/z_lib_f$F/plan4/Z3 ] && [ -d $H/data/geom_plan4_f$F ] \
    && log "fold $F data ready" || log "fold $F data NOT ready"
done
ready() { [ -s $H/data/clip_split_cv/fold$1/clips_train_xzmix.txt ] && [ -d $H/data/z_lib_f$1/plan4/Z3 ]; }
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # fold P gpu seed
  local F=$1 P=$2 G=$3 s=$4 VS=$V
  [ $F != 1 ] && VS=${VS}_f$F
  local S=$H/runs/students_${VS}_s$s O=$H/runs/students_${VS}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P LT=$H/data/z_tables_cv_f${F}_xzmix/$P
  local ALLOW=$H/data/clip_split_cv/fold$F/clips_train_xzmix.txt AUG=$(aug $F) SPL=$H/data/clip_split_cv/fold$F/split.json
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${VS}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F $P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${VS}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F $P s$s round 2 failed"; return; }
  # condition A: own z, human clock (pre-registered)
  for CR in task track; do
    local R=$H/runs/results_${V}$([ $CR = track ] && echo trk)$([ $F != 1 ] && echo _f$F)
    for DRV in 0 1; do for X in "" "--push"; do
      [ -f ${R}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
      HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
        --student $O/$P/student_2000.pt --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
    done; done
  done
  # condition B: library z + planner (plan4), deployable
  local RB=$H/runs/results_${V}B$([ $F != 1 ] && echo _f$F) GP=$H/data/geom_plan4_f$F
  for DRV in 0 1; do for X in "" "--push"; do
    [ -f ${RB}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
    HNU_GEOM_W1=$GP/ref_stakes_s93_h10 HNU_GEOM_W2=$GP/ref_stakes_s83_h10 HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=task HNU_DR=$DRV \
      HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_lib_f$F/plan4/$P --per-clip 32 $X --out ${RB}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
  done; done
  log "fold $F $P s$s done"
}
JOBS=()
for s in 1 2 3; do for F in 1 2 3 4; do for P in Z0 Z3 Z1 Z2; do JOBS+=("$F $P $s"); done; done; done  # main (48)
for s in 4 5; do JOBS+=("1 Z3 $s"); done                                                             # deployment seeds
Q=$N/${V}_claims
rm -rf $Q; mkdir -p $Q
worker() {  # gpu
  while true; do
    local left=0 took=0
    for i in "${!JOBS[@]}"; do
      [ -d $Q/$i ] && continue
      left=1
      set -- ${JOBS[$i]}
      ready $1 || continue
      mkdir $Q/$i 2> /dev/null || continue
      one $1 $2 $G $3
      took=1
      break
    done
    [ $left = 0 ] && return
    [ $took = 0 ] && sleep 300
  done
}
for G in $(seq 0 $((NGPU - 1))); do G=$G worker & sleep 20; done
wait
for j in "${JOBS[@]}"; do set -- $j; one $1 $2 0 $3; done  # retry failures (sequential, GPU 0)
n=$(ls $H/runs/students_${V}_d2_s*/Z*/student_2000.pt $H/runs/students_${V}_f*_d2_s*/Z*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of ${#JOBS[@]}"
[ "$n" = ${#JOBS[@]} ] && log "finished"
