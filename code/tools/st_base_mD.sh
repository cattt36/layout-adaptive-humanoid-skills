#!/bin/bash
# real brace dims measured by the user (photos): brace axis meets the post ~33 cm above the bar top (~36 cm
# above the floor), foot ~10.5 cm from the bar end (39 cm from the post axis), tube ~3 cm.  Same S2k policy, scene
# re-exported with these dims; sim2sim on the 9 planned layouts (A / B / C) -> robot_sim_check/out_base_S2k_mD
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; RS=/root/gpufree-data/robot_sim_check
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
E=$H/deploy_exports/v6g10xm_base_S2k_s5_mD/L3; mkdir -p $E
[ -f $E/policy.onnx ] || HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.03,0.39,0.36,0.03 CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py \
  --student $H/runs/students_v6g10xm_base_S2k_s5/Z3/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
OUT=$RS/out_base_S2k_mD; mkdir -p $OUT
for L in $RS/v6/in/planned_layouts/*/; do
  L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
  for c in A B C; do
    X=""; case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
    [ -f $OUT/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 100 --tasks $t \
      --motion-dirs $L --latent-dir $L $X --out $OUT/${n}_$c.json > $OUT/${n}_$c.log 2>&1
  done
done
echo done > $OUT/DONE
