#!/bin/bash
# (user plan): cut the floor bars of the two INNER bridge posts short and re-weld shorter braces; which bar
# length per side lets the deployed model (Deployment_C, trained WITHOUT bases) pass?  Outer posts keep the
# measured bases (bar 100 x 10 x 4 cm, braces foot 39 cm / top 37 cm, tube 3 cm).  Inner posts: bar half-length L from
# the post axis, brace foot 5 cm in from the bar end, ~45 deg (top height = foot distance), HNU_BASE_MID (deploy/sim2sim.py).
# CPU sim2sim on D1 / D2 / N3, conditions A / B / C, 100 trials.  Out: robot_sim_check/out_mid_old/<case>/
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_mid_old
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
E=$H/deploy_exports/v6g10xm_f1_s5_base_mDh40/L3; mkdir -p $E
[ -f $E/policy.onnx ] || HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py \
  --student $H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
for CASE in none 0.10 0.15 0.20 0.25 0.30 0.35 0.40 full; do
  case $CASE in
    full) MID="";;
    none) MID='{"none": true}';;
    *) MID=$($PYD -c "L=$CASE; d=max(L-0.05,0.06); print('{\"bar_len\": %.3f, \"brace_d\": %.3f, \"brace_h\": %.3f}' % (2*L, d, d))");;
  esac
  mkdir -p $O/$CASE; echo "$MID" > $O/$CASE/mid.json
  for L in $RS/v6/in/planned_layouts/[DN]*Crouched*/; do
    L=${L%/}; n=$(basename $L)
    for c in A B C; do
      X=""; case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
      [ -f $O/$CASE/${n}_$c.json ] || HNU_BASE_MID="$MID" $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 \
        --workers 100 --tasks CrouchedTraverse --motion-dirs $L --latent-dir $L $X --out $O/$CASE/${n}_$c.json > $O/$CASE/${n}_$c.log 2>&1
    done
  done
done
echo done > $O/DONE
