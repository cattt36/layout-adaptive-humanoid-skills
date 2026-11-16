#!/bin/bash
# L40, (user choice B): second pilot round of the cross-trial z augmentation; recipe as tools/xz_l40.sh.
#   GPU 0 / 1 : L0 / L3, 25 % mismatched (make_xz.py 1 25)              -> v6g10x25, seed 1
#   GPU 2 / 3 : L0 / L3, 50 % mismatched, half linear / half aligned (mix) -> v6g10xm,  seed 1
#   GPU 4     : L3, linear 50 % (as v6g10x) with RL seed 2                 -> v6g10x,   seed 2 (seed variability)
# Evaluation as xz_l40.sh (+ plan4); the old v6g10n seed-2 L3 is evaluated under the same conditions for GPU 4.
# Log: runs/night/xz_r2.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # data mode
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_$1_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_$1_f1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_$1_f1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/xz_r2.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
log "prepare: $CK"
[ -s $H/data/clip_split_cv/fold1/clips_train_xz25.txt ] || $PY tools/make_xz.py 1 25 >> $N/xz_r2.log 2>&1
[ -s $H/data/clip_split_cv/fold1/clips_train_xzmix.txt ] || $PY tools/make_xz.py 1 mix >> $N/xz_r2.log 2>&1
train() {  # P data-mode student-name seed gpu
  local P=$1 M=$2 V=$3 s=$4 G=$5
  local S=$H/runs/students_${V}_s$s O=$H/runs/students_${V}_d2_s$s LT=$H/data/z_tables_cv_f1_$M/$P
  local ALLOW=$H/data/clip_split_cv/fold1/clips_train_$M.txt AUG=$(aug $M)
  log "$V s$s $P: round 1 start (gpu $G, data $M)"
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$V s$s $P round 1 failed"; return 1; }
  log "$V s$s $P: round 2 start"
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$V s$s $P round 2 failed"; return 1; }
  log "$V s$s $P: trained"
}
ev() {  # condition tag student seed gpu geom-root latent-dir
  local O=$H/runs/results_xz_$1_s$4_dr0
  [ -f $O/eval_student_$2.json ] && return
  env ${6:+HNU_GEOM_W1=$6/ref_stakes_s93_h10 HNU_GEOM_W2=$6/ref_stakes_s83_h10} HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json \
    HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$5 $PY eval_student.py --student $3 \
    --latent-dir $7 --per-clip 32 --out $O --tag _$2 >> $N/xz_r2_eval.log 2>&1
  log "eval $1 s$4 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
}
evals() {  # P tag-prefix student-path seed gpu
  local P=$1 T=$2_$1 ST=$3 s=$4 G=$5 G3=$H/data/geom_plan3_f1 G4=$H/data/geom_plan4_f1
  ev own $T $ST $s $G "" $H/data/z_tables_cv_f1/$P
  ev geo $T $ST $s $G "" $H/data/z_lib_f1/geo/$P
  ev plan3 $T $ST $s $G $G3 $H/data/z_lib_f1/plan3/$P
  ev plan4 $T $ST $s $G $G4 $H/data/z_lib_f1/plan4/$P
  ev geoplan3 $T $ST $s $G $G3 $H/data/z_lib_f1/geo/$P
  ev ownplan3 $T $ST $s $G $G3 $H/data/z_tables_cv_f1/$P
}
run() {  # P data-mode student-name tag seed gpu
  train $1 $2 $3 $5 $6 && evals $1 $4 $H/runs/students_$3_d2_s$5/$1/student_2000.pt $5 $6
}
log "start"
( evals Z3 n $H/runs/students_v6g10n_d2_s2/Z3/student_2000.pt 2 4 ) &
( run Z0 xz25 v6g10x25 x25 1 0 ) & ( run Z3 xz25 v6g10x25 x25 1 1 ) &
( run Z0 xzmix v6g10xm xm 1 2 ) & ( run Z3 xzmix v6g10xm xm 1 3 ) & ( sleep 300; run Z3 xz v6g10x x 2 4 ) &
wait
log "done"
