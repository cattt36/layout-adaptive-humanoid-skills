#!/bin/bash
# (user question: do wider stakes help S1 / N1, where the robot lands on the stake edge?): stake radius 15 / 20 /
# 25 cm (height 10 cm) with the current deployment (S2k student, measured post bases, = Deployment_D).  Scenes exported
# per radius (r15 = the deployed export); CPU sim2sim on the 6 planned stake layouts, conditions A / C, 100 trials,
# criterion v2; condition A also saves state traces (foot-on-stake analysis: tools/stake_edge_summary.py).
# Out: robot_sim_check/out_stakesize_s2k/r<cm>/<layout>_<cond>.json (+ trace_<layout>/)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_stakesize_s2k
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
for R in 15 20 25; do
  ( if [ $R = 15 ]; then E=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3
    else
      E=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40_r$R/L3; mkdir -p $E
      [ -f $E/policy.onnx ] || HNU_STAKE_R=0.$R HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 CUDA_VISIBLE_DEVICES=$((R / 10)) \
        $PY deploy/export_student.py --student $H/runs/students_v6g10xm_base_S2k_s5/Z3/student_2000.pt \
        --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
      [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
    fi
    mkdir -p $O/r$R
    for L in $RS/v6/in/planned_layouts/[STN][12]_*Step*; do
      n=$(basename $L); t=$(echo $n | cut -d_ -f2)
      for c in A C; do
        X=""; TR=""
        case $c in A) TR=$O/r$R/trace_$n; mkdir -p $TR;;
                   C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
        [ -f $O/r$R/${n}_$c.json ] || HNU_S2S_TRACE=$TR $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 40 \
          --tasks $t --motion-dirs $L --latent-dir $L $X --out $O/r$R/${n}_$c.json > $O/r$R/${n}_$c.log 2>&1
      done
    done ) &
done
wait
echo ALLDONE > $O/DONE
