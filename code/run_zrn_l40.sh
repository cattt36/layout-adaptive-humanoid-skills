#!/bin/bash
# Explicit-reference baseline WITHOUT rhythm decoupling ("ER-conv", level ZR in the v6g10n names), L40, 
# (user decision, reviewer-proofing: completes the 2x2 interface x training design; analysis plan fixed before the runs
# in runs/zrn_analysis_plan.md).  Identical to run_zr_l40.sh / run_zr_l40_s23.sh (ER with decoupling: same teachers,
# ZR tables, DAgger 4000 + 2000, seeds 1000*s / 2000*s, z offsets, network, randomization, evaluation) except the
# training data, which are those of the conventional students of run_v6g10n_l40b.sh: training trials + tempo copies only
# (clips_train_aug.txt, aug_ts_f<k> geometry, z_tables_cv_f<k>_aug/ZR), no cross-trial skill augmentation.
# 12 jobs (folds 1-4 x RL seeds 1-3, seed-major) on the 5 GPUs.  Results in the conventional result dirs with tag _ZR:
# condition A results_v6g10n[trk][_f<k>]_s<s>_dr<0|1>, condition B results_v6g10nB[_f<k>]_s<s>_dr<0|1>.
# Log runs/night/zrn.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10n
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold (identical to run_v6g10n_l40b.sh: tempo copies only)
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_ts_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/zrn.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
log "start: teachers $CK"
one() {  # fold gpu seed
  local F=$1 G=$2 s=$3 P=ZR VS=$V
  [ $F != 1 ] && VS=${VS}_f$F
  local S=$H/runs/students_${VS}_s$s O=$H/runs/students_${VS}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P LT=$H/data/z_tables_cv_f${F}_aug/$P
  local ALLOW=$H/data/clip_split_cv/fold$F/clips_train_aug.txt AUG=$(aug $F) SPL=$H/data/clip_split_cv/fold$F/split.json
  log "fold $F s$s: start on gpu $G"
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${VS}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F s$s round 1 FAILED"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${VS}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F s$s round 2 FAILED"; return; }
  log "fold $F s$s trained"
  for CR in task track; do  # condition A (own reference, demonstrated clock)
    local R=$H/runs/results_${V}$([ $CR = track ] && echo trk)$([ $F != 1 ] && echo _f$F)
    for DRV in 0 1; do for X in "" "--push"; do
      [ -f ${R}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
      HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
        --student $O/$P/student_2000.pt --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
    done; done
  done
  local RB=$H/runs/results_${V}B$([ $F != 1 ] && echo _f$F) GP=$H/data/geom_plan4_f$F
  for DRV in 0 1; do for X in "" "--push"; do  # condition B (nearest library reference + planner plan4)
    [ -f ${RB}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
    HNU_GEOM_W1=$GP/ref_stakes_s93_h10 HNU_GEOM_W2=$GP/ref_stakes_s83_h10 HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=task HNU_DR=$DRV \
      HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_lib_f$F/plan4/$P --per-clip 32 $X --out ${RB}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
  done; done
  touch $O/$P/student_2000.pt.ok
  log "fold $F s$s trained and evaluated"
}
JOBS=()
for s in 1 2 3; do for F in 1 2 3 4; do JOBS+=("$F $s"); done; done
Q=$N/zrn_claims
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
n=$(ls $H/runs/students_${V}_d2_s*/ZR/student_2000.pt.ok $H/runs/students_${V}_f*_d2_s*/ZR/student_2000.pt.ok 2>/dev/null | wc -l)
log "all jobs complete: $n of ${#JOBS[@]} students trained and evaluated"
