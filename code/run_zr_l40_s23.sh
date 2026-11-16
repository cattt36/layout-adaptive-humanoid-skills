#!/bin/bash
# Fair baseline ZR, RL seeds 2 and 3 (user decision: same seeds as the main study so that ZR enters the
# 8.2 table directly).  Identical to run_zr_l40.sh (seed 1) except the seed; ZR tables already exist.  Starts when the
# old-student completion (tools/old_complete_l40_now.sh) has ended; 8 jobs (folds 1-4 x seeds 2-3) on the 5 GPUs.
# Log runs/night/zr.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10xm
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold (identical to run_v6g10xm_l40.sh / run_zr_l40.sh)
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_xzmix_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_xzmix_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_xzmix_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/zr.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
for i in $(seq 1 600); do
  grep -q "old_complete: all old students trained and evaluated under condition B" $N/v6g10n.log && break
  sleep 120
done
log "ZR seeds 2-3: start"
one() {  # fold gpu seed -- body of run_zr_l40.sh one() with the seed as argument
  local F=$1 G=$2 s=$3 P=ZR VS=$V
  [ $F != 1 ] && VS=${VS}_f$F
  local S=$H/runs/students_${VS}_s$s O=$H/runs/students_${VS}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P LT=$H/data/z_tables_cv_f${F}_xzmix/$P
  local ALLOW=$H/data/clip_split_cv/fold$F/clips_train_xzmix.txt AUG=$(aug $F) SPL=$H/data/clip_split_cv/fold$F/split.json
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${VS}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F ZR s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${VS}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F ZR s$s round 2 failed"; return; }
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
  log "fold $F ZR s$s trained and evaluated"
}
JOBS=("1 2" "2 2" "3 2" "4 2" "1 3" "2 3" "3 3" "4 3")
Q=$N/zr_s23_claims
rm -rf $Q; mkdir -p $Q
worker() {
  for i in "${!JOBS[@]}"; do
    mkdir $Q/$i 2> /dev/null || continue
    set -- ${JOBS[$i]}
    one $1 $G $2
  done
}
for G in 0 1 2 3 4; do G=$G worker & sleep 20; done
wait
log "ZR seeds 2-3: all folds complete"
