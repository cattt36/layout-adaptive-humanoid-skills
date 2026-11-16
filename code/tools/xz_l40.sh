#!/bin/bash
# L40 server, ~15:10 (user decision: the v6g10n queue is stopped, all GPUs on the new method).
# Students with cross-trial z augmentation (tools/make_xz.py), recipe identical to run_v6g10n_l40b.sh one() except the
# training data.  Fold 1, RL seed 1:
#   GPU 0-1, 3-4 : L3 / L0 / L1 / L2, mode xz  (library z linearly resampled; z and clock disagree)   -> v6g10x
#   GPU 2        : L3, mode xzw (library z warped to the clip's own stance windows; z and clock agree) -> v6g10xw
# Evaluation (nominal, task criterion, split_v2 test clips, 32 rollouts), as tools/xz_pilot.sh:
#   own (own z, human clock) / geo (library z linear) / plan3 (library z warped to the planned clock + planned clock) /
#   geoplan3 (library z linear + planned clock) / ownplan3 (own z + planned clock); old v6g10n students under geoplan3.
# Log: runs/night/xz_l40.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # mode
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_$1_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_$1_f1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_$1_f1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/xz_l40.log; }
CK=$(cat $N/v6g10n_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
s=1
log "prepare: $CK"
[ -s $H/data/clip_split_cv/fold1/clips_train_xz.txt ] || $PY tools/make_xz.py 1 >> $N/xz_l40.log 2>&1
[ -s $H/data/clip_split_cv/fold1/clips_train_xzw.txt ] || $PY tools/make_xz.py 1 w >> $N/xz_l40.log 2>&1
[ -d $H/data/z_lib_f1/geo/Z3 ] || $PY tools/make_z_library.py 1 >> $N/xz_l40.log 2>&1
[ -d $H/data/z_lib_f1/plan3/Z3 ] || $PY tools/make_z_plan.py 1 plan3 >> $N/xz_l40.log 2>&1
train() {  # P mode gpu
  local P=$1 M=$2 G=$3 V=v6g10x; [ $2 = xzw ] && V=v6g10xw
  local S=$H/runs/students_${V}_s$s O=$H/runs/students_${V}_d2_s$s LT=$H/data/z_tables_cv_f1_$M/$P
  local ALLOW=$H/data/clip_split_cv/fold1/clips_train_$M.txt AUG=$(aug $M)
  log "$V $P: round 1 start (gpu $G)"
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LT --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$V $P round 1 failed"; return 1; }
  log "$V $P: round 2 start"
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$V $P round 2 failed"; return 1; }
  log "$V $P: trained"
}
ev() {  # variant tag student gpu geom-root latent-dir
  local O=$H/runs/results_xz_$1_s${s}_dr0
  [ -f $O/eval_student_$2.json ] && return
  env ${5:+HNU_GEOM_W1=$5/ref_stakes_s93_h10 HNU_GEOM_W2=$5/ref_stakes_s83_h10} HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json \
    HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$4 $PY eval_student.py --student $3 \
    --latent-dir $6 --per-clip 32 --out $O --tag _$2 >> $N/xz_l40_eval.log 2>&1
  log "eval $1 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
}
evals() {  # P mode gpu
  local P=$1 G=$3 V=v6g10x; [ $2 = xzw ] && V=v6g10xw
  local NEW=$H/runs/students_${V}_d2_s$s/$P/student_2000.pt T=${V#v6g10}_$P GP=$H/data/geom_plan3_f1
  ev own $T $NEW $G "" $H/data/z_tables_cv_f1/$P
  ev geo $T $NEW $G "" $H/data/z_lib_f1/geo/$P
  ev plan3 $T $NEW $G $GP $H/data/z_lib_f1/plan3/$P
  ev geoplan3 $T $NEW $G $GP $H/data/z_lib_f1/geo/$P
  ev ownplan3 $T $NEW $G $GP $H/data/z_tables_cv_f1/$P
}
old() {  # reference: the v6g10n students (L40) under the same conditions
  local G=$1 GP=$H/data/geom_plan3_f1
  for P in Z0 Z1 Z2 Z3; do
    local OLD=$H/runs/students_v6g10n_d2_s$s/$P/student_2000.pt
    ev own n_$P $OLD $G "" $H/data/z_tables_cv_f1/$P
    ev geo n_$P $OLD $G "" $H/data/z_lib_f1/geo/$P
    ev plan3 n_$P $OLD $G $GP $H/data/z_lib_f1/plan3/$P
    ev geoplan3 n_$P $OLD $G $GP $H/data/z_lib_f1/geo/$P
    ev ownplan3 n_$P $OLD $G $GP $H/data/z_tables_cv_f1/$P
  done
}
log "start"
( old 4 ) &
( train Z3 xz 0 && evals Z3 xz 0 ) & ( train Z0 xz 1 && evals Z0 xz 1 ) & ( train Z3 xzw 2 && evals Z3 xzw 2 ) &
( train Z1 xz 3 && evals Z1 xz 3 ) & ( sleep 600; train Z2 xz 4 && evals Z2 xz 4 ) &
wait
log "done"
