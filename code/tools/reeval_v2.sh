#!/bin/bash
# Re-evaluation under success criterion v2: teacher bank + every finished v6g10m student,
# 32 rollouts per clip, nominal / randomised x clean / push.  Output: results_v6g10mv2_*  (pipeline results untouched)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_START_FRAC=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
R=$H/runs/results_v6g10mv2
JOBS=("teacher 0")
for d in $H/runs/students_v6g10m_d2_s*; do
  s=${d##*_s}
  for P in Z0 Z1 Z2 Z3 Z4 NL REF; do [ -f $d/$P/student_2000.pt ] && JOBS+=("$P $s"); done
done
one() {  # P seed gpu
  local P=$1 s=$2 G=$3
  for DRV in 0 1; do for X in "" "--push"; do
    if [ $P = teacher ]; then
      O=${R}_bank_teacher_dr$DRV; F=$O/eval_teacher_teacher$([ -n "$X" ] && echo _push).json
      [ -f $F ] || HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom \
        --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 $X --out $O --tag _teacher \
        > /dev/null 2>&1
    else
      LD=$H/data/z_tables_v5t/$P; [ $P = NL ] || [ $P = REF ] && LD=$H/data/z_tables_v5t/Z0
      O=${R}_s${s}_dr$DRV; F=$O/eval_student_$P$([ -n "$X" ] && echo _push).json
      [ -f $F ] || HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
        --student $H/runs/students_v6g10m_d2_s$s/$P/student_2000.pt --latent-dir $LD --per-clip 32 $X --out $O --tag _$P \
        > /dev/null 2>&1
    fi
  done; done
  echo "$P s$s done"
}
Q=$H/runs/reeval_v2_queue; echo 0 > $Q.next; : > $Q.lock
worker() {
  while true; do
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    one ${JOBS[$i]} $1
  done
}
worker 0 & worker 1 & wait
$PY tools/table_v2.py
