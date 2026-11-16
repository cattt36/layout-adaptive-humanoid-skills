#!/bin/bash
# 4090 server, (user request; exploratory): layout-adapted skill with a planned clock (tools/make_z_plan.py).
# Same protocol as tools/z_library.sh (fold-1 students of RL seed $SEED, nominal, criterion task / v2, 32 rollouts per
# clip), with the stake geometry replaced by data/geom_plan_f1 (planned stance windows):
#   plan     : library z warped to the planned windows
#   ownclock : the clip's own z under the planned clock (control: is the planned clock itself executable?)
# Log: runs/night/z_plan.log;  table: runs/z_plan_s<seed>.md (tools/z_plan_table.py)
# VAR=plan2: initial wait capped at 0.5 s (make_z_plan.py); ownclock2 = own z under that clock.
SEED=${SEED:-1}
VAR=${VAR:-plan}
OC=ownclock; [ $VAR = plan ] || OC=ownclock${VAR#plan}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/geom_${VAR}_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/geom_${VAR}_f1/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 \
  HNU_POST_SHAPE=mesh12 HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/z_plan.log; }
S=$H/runs/students_v6g10n_d2_s$SEED
log "start seed $SEED $VAR"
$PY tools/make_z_plan.py 1 $VAR >> $N/z_plan.log 2>&1 || { log "STOP: make_z_plan failed"; exit 1; }
lane() {
  local G=$1; shift
  for j in "$@"; do
    set -- $j
    local O=$H/runs/results_v6g10n_zlib_$1_s${SEED}_dr0 L=$H/data/z_lib_f1/$1/$2
    [ $1 = $OC ] && L=$H/data/z_tables_cv_f1/$2
    [ -f $O/eval_student_$2.json ] && continue
    HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $S/$2/student_2000.pt \
      --latent-dir $L --per-clip 32 --out $O --tag _$2 >> $N/z_plan_eval.log 2>&1
    log "$1 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
  done
}
lane 0 "$VAR Z0" "$VAR Z2" "$OC Z0" "$OC Z2" & lane 1 "$VAR Z3" "$VAR Z1" "$OC Z3" "$OC Z1" & wait
$PY tools/z_plan_table.py $SEED > $H/runs/z_plan_s$SEED.md 2>> $N/z_plan.log
log "done -> runs/z_plan_s$SEED.md"
