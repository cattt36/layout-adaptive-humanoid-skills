#!/bin/bash
# Baseline for v6r2: the v5 + DAgger-2 bridge students (RL seeds 1-2) evaluated in the randomised environment
# (HNU_DR=1, full level), clean and push.  Same protocol as the v6r2 robustness test.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_DR=1 HNU_DR_LEVEL=2
for s in 1 2; do
  S=$H/runs/students_v5_d2; [ $s = 2 ] && S=$H/runs/students_v5_d2_s2
  for P in Z0 Z1 Z2 Z3 Z4; do
    for X in "" "--push"; do
      CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --student $S/$P/student_2000.pt --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X \
        --out $H/runs/results_v5_d2_s${s}_dr1 --tag _$P >> $H/runs/night/v5_dr1_eval.log 2>&1
    done
  done
done
$PY tools/bridge_err.py $H/runs/results_v5_d2_s1_dr1 $H/runs/results_v5_d2_s2_dr1 > $H/runs/night/v5_dr1_summary.txt 2>&1
