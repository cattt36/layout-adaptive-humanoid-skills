#!/bin/bash
# Tempo check: can the TEACHER bank (and, for reference, the v6g10j seed-1 L0 / L4 students) follow references
# played 0.85x / 0.9x / 1.1x / 1.15x?  Task criterion, nominal and randomised, 16 rollouts per clip.
# Teacher trials are reported on train trials (the teachers saw every trial at 1.0x); students on train and test.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
$PY tools/make_timescale.py 0.85 0.9 1.1 1.15 || exit 1
export HF_HUB_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_START_FRAC=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
run_speed() {  # speed tag, gpu
  local T=$1 G=$2 R=$H/data/ts/ts$1 O=$H/runs/ts_check/ts$1
  for DRV in 0 1; do
    HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift HNU_DR=$DRV \
      CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
      --latent-dir $R/z_tables_v5t/Z4 --per-clip 16 --out $O/dr$DRV --tag _teacher > $O.teacher$DRV.log 2>&1 || tail -3 $O.teacher$DRV.log
    for P in Z0 Z4; do
      HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift HNU_DR=$DRV \
        CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10j_d2_s1/$P/student_2000.pt \
        --latent-dir $R/z_tables_v5t/$P --per-clip 16 --out $O/dr$DRV --tag _$P > $O.$P$DRV.log 2>&1 || tail -3 $O.$P$DRV.log
    done
  done
}
mkdir -p $H/runs/ts_check
(run_speed 085 0; run_speed 110 0) & (run_speed 090 1; run_speed 115 1) & wait
$PY tools/ts_table.py
