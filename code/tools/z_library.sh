#!/bin/bash
# 4090 server, (user request; exploratory): skill library on a new layout.  Fold-1 students of RL seed $SEED
# (default 1) evaluated on every clip with the z of a library clip (tools/make_z_library.py: random / geo / geowarp),
# nominal condition, criterion task / v2, 32 rollouts per clip -- same protocol as the 4090 reference
# results_v6g10n_C0_s<seed>_dr0 (own z) and the z ablation (results_v6g10n_zabl_*).  No student is changed.
# Log: runs/night/z_library.log;  table: runs/z_library_s<seed>.md (tools/z_library_table.py)
SEED=${SEED:-1}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/z_library.log; }
S=$H/runs/students_v6g10n_d2_s$SEED
[ $(ls $S/Z*/student_2000.pt 2>/dev/null | wc -l) = 5 ] || { log "STOP: students missing"; exit 1; }
log "start seed $SEED (students $(md5sum $S/Z*/student_2000.pt | md5sum | cut -c1-8))"
[ -d $H/data/z_lib_f1/geowarp/Z4 ] || $PY tools/make_z_library.py 1 >> $N/z_library.log 2>&1
lane() {  # gpu, jobs "variant level"
  local G=$1; shift
  for j in "$@"; do
    set -- $j
    local O=$H/runs/results_v6g10n_zlib_$1_s${SEED}_dr0
    [ -f $O/eval_student_$2.json ] && continue
    HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $S/$2/student_2000.pt \
      --latent-dir $H/data/z_lib_f1/$1/$2 --per-clip 32 --out $O --tag _$2 >> $N/z_library_eval.log 2>&1
    log "$1 $2: $([ -f $O/eval_student_$2.json ] && echo ok || echo FAILED)"
  done
}
J=(); for v in geo geowarp random; do for P in Z0 Z4 Z1 Z2 Z3; do J+=("$v $P"); done; done
A=(); B=(); for i in "${!J[@]}"; do [ $((i % 2)) = 0 ] && A+=("${J[$i]}") || B+=("${J[$i]}"); done
lane 0 "${A[@]}" & lane 1 "${B[@]}" & wait
$PY tools/z_library_table.py $SEED > $H/runs/z_library_s$SEED.md 2>> $N/z_library.log
log "finished -> runs/z_library_s$SEED.md"
