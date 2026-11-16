#!/bin/bash
# 4090 server, (user decision; exploratory pilot): students with cross-trial z augmentation (tools/make_xz.py).
# Recipe identical to run_v6g10n_l40b.sh one() (teachers, augmentation, DAgger 4000 + 2000) except the training data
# (aug_xz_f1 / z_tables_cv_f1_xz / clips_train_xz.txt).  Fold 1, RL seed 1, L0 and L3 (one GPU each).  NOTE: trained on
# the 4090 (the L40 is busy with the primary study) -- hardware differs from the v6g10n students; confirmatory runs
# would go to the L40.  Evaluation (nominal, task criterion, test clips of split_v2, 32 rollouts):
#   own      : own z, human clock (policy level, must not drop)
#   geo      : nearest library z, linear (data/z_lib_f1/geo)
#   plan3    : nearest library z warped to the planned clock + planned clock (data/z_lib_f1/plan3, geom_plan3_f1)
#   geoplan3 : nearest library z, linear, + planned clock (the new student should take the timing from the clock)
# Log: runs/night/xz_pilot.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_xz_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_xz_f1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_xz_f1/ref_bridge_video_lift"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/xz_pilot.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
V=v6g10x s=1
[ -s $H/data/clip_split_cv/fold1/clips_train_xz.txt ] || $PY tools/make_xz.py 1 >> $N/xz_pilot.log 2>&1
[ -d $H/data/z_lib_f1/plan3/Z3 ] || $PY tools/make_z_plan.py 1 plan3 >> $N/xz_pilot.log 2>&1
train() {  # P gpu
  local P=$1 G=$2 S=$H/runs/students_${V}_s$s O=$H/runs/students_${V}_d2_s$s LT=$H/data/z_tables_cv_f1_xz/$1
  local ALLOW=$H/data/clip_split_cv/fold1/clips_train_xz.txt
  log "$P: round 1 start (gpu $G)"
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P round 1 failed"; return; }
  log "$P: round 2 start"
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P round 2 failed"; return; }
  log "$P: trained"
}
ev() {  # variant student-tag latent-dir gpu [geom-root]
  local O=$H/runs/results_xzpilot_$1_s${s}_dr0
  [ -f $O/eval_student_$2.json ] && return
  env ${5:+HNU_GEOM_W1=$5/ref_stakes_s93_h10 HNU_GEOM_W2=$5/ref_stakes_s83_h10} HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json \
    HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$4 $PY eval_student.py --student $3 \
    --latent-dir $6 --per-clip 32 --out $O --tag _$2 >> $N/xz_pilot_eval.log 2>&1
  log "eval $1 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
}
evals() {  # P gpu: new student (tag P) and, for reference, the v6g10n student under the planned-clock conditions
  local P=$1 G=$2 NEW=$H/runs/students_${V}_d2_s$s/$1/student_2000.pt OLD=$H/runs/students_v6g10n_d2_s$s/$1/student_2000.pt
  local GP=$H/data/geom_plan3_f1
  ev own $P $NEW $G "" $H/data/z_tables_cv_f1/$P
  ev geo $P $NEW $G "" $H/data/z_lib_f1/geo/$P
  ev plan3 $P $NEW $G $GP $H/data/z_lib_f1/plan3/$P
  ev geoplan3 $P $NEW $G $GP $H/data/z_lib_f1/geo/$P
  ev ownplan3 $P $NEW $G $GP $H/data/z_tables_cv_f1/$P
  ev geoplan3 old$P $OLD $G $GP $H/data/z_lib_f1/geo/$P
}
log "start: $CK"
( train Z3 0; evals Z3 0 ) & ( train Z0 1; evals Z0 1 ) & wait
log "done"
