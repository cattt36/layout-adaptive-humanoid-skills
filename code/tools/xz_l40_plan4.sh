#!/bin/bash
# L40, after tools/xz_l40.sh, evaluate every new (v6g10x / v6g10xw) and old (v6g10n) fold-1 seed-1 student
# under "plan4" (plan3 + balance-beam entry planning, tools/make_z_plan.py plan4).  Same protocol as xz_l40.sh ev().
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_GEOM_W1=$H/data/geom_plan4_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/geom_plan4_f1/ref_stakes_s83_h10
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/xz_l40.log; }
while pgrep -fx "bash tools/xz_l40.sh" > /dev/null; do sleep 60; done
J=()
for P in Z0 Z1 Z2 Z3; do J+=("n_$P $H/runs/students_v6g10n_d2_s1/$P/student_2000.pt $P" "x_$P $H/runs/students_v6g10x_d2_s1/$P/student_2000.pt $P"); done
J+=("xw_Z3 $H/runs/students_v6g10xw_d2_s1/Z3/student_2000.pt Z3")
lane() {
  local G=$1; shift
  for j in "$@"; do
    set -- $j
    local O=$H/runs/results_xz_plan4_s1_dr0
    [ -f $O/eval_student_$1.json ] && continue
    [ -f $2 ] || { log "plan4 $1: student missing"; continue; }
    HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G \
      $PY eval_student.py --student $2 --latent-dir $H/data/z_lib_f1/plan4/$3 --per-clip 32 --out $O --tag _$1 >> $N/xz_l40_eval.log 2>&1
    log "eval plan4 $1: $([ -f $O/eval_student_$1.json ] && echo ok || echo FAILED)"
  done
}
A=(); B=(); C=(); for i in "${!J[@]}"; do case $((i % 3)) in 0) A+=("${J[$i]}");; 1) B+=("${J[$i]}");; 2) C+=("${J[$i]}");; esac; done
lane 0 "${A[@]}" & lane 1 "${B[@]}" & lane 2 "${C[@]}" & wait
log "plan4 evaluations done"
