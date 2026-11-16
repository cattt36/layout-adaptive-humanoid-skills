#!/bin/bash
# Fair external baseline on the L40 (user decision; the only baseline reported in the paper): the explicit
# reference student, level "ZR" (tools/make_zr.py), trained and evaluated exactly like the v6g10xm students of
# run_v6g10xm_l40.sh -- same teachers, deployable observations, fold training trials, cross-trial "mix" recipe and z
# augmentation, DAgger 4000 + 2000, z offsets 0..50 -- but reading the standardised reference command (joint positions /
# velocities) instead of z_skill.  Folds 1-4, RL seed 1, one job per GPU 0-3 next to the running main study.
# Evaluation as the main study: condition A (own reference, human clock; task / track x DR 0/1 x push), condition B
# (reference of the nearest library trial + planner plan4).  Results in the main result dirs with tag _ZR.
# Log runs/night/zr.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10xm
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold (identical to run_v6g10xm_l40.sh)
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_xzmix_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_xzmix_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_xzmix_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/zr.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
log "start: teachers $CK"
for F in 1 2 3 4; do
  { $PY tools/make_zr.py $F && HNU_LEVELS=ZR $PY tools/make_z_library.py $F && \
    HNU_LEVELS=ZR HNU_SKIP_GEOM=1 $PY tools/make_z_plan.py $F plan4; } >> $N/zr.log 2>&1 \
    && log "fold $F ZR tables ready" || { log "fold $F ZR tables FAILED"; exit 1; }
done
one() {  # fold gpu (seed 1, level ZR) -- body of run_v6g10xm_l40.sh one()
  local F=$1 G=$2 s=1 P=ZR VS=$V
  [ $F != 1 ] && VS=${VS}_f$F
  local S=$H/runs/students_${VS}_s$s O=$H/runs/students_${VS}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P LT=$H/data/z_tables_cv_f${F}_xzmix/$P
  local ALLOW=$H/data/clip_split_cv/fold$F/clips_train_xzmix.txt AUG=$(aug $F) SPL=$H/data/clip_split_cv/fold$F/split.json
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${VS}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F ZR round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${VS}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F ZR round 2 failed"; return; }
  log "fold $F ZR trained"
  for CR in task track; do
    local R=$H/runs/results_${V}$([ $CR = track ] && echo trk)$([ $F != 1 ] && echo _f$F)
    for DRV in 0 1; do for X in "" "--push"; do
      [ -f ${R}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
      HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
        --student $O/$P/student_2000.pt --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
    done; done
  done
  local RB=$H/runs/results_${V}B$([ $F != 1 ] && echo _f$F) GP=$H/data/geom_plan4_f$F
  for DRV in 0 1; do for X in "" "--push"; do
    [ -f ${RB}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
    HNU_GEOM_W1=$GP/ref_stakes_s93_h10 HNU_GEOM_W2=$GP/ref_stakes_s83_h10 HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=task HNU_DR=$DRV \
      HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_lib_f$F/plan4/$P --per-clip 32 $X --out ${RB}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
  done; done
  log "fold $F ZR evaluated"
}
one 1 0 & sleep 30; one 2 1 & sleep 30; one 3 2 & sleep 30; one 4 3 & wait
log "ZR all folds complete"
