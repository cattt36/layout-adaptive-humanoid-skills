#!/bin/bash
# (user: Deployment_H_mocap first): post-processing of the stakes student with the mocap velocity
# (training started by tools/rtH_vel.sh, its shell stopped, the training itself keeps running).  Stakes layouts and
# stakes candidates only (the bridge model of the bundle is rtGm c1000); c1000 as soon as student_1000.pt exists,
# then c2000.  Afterwards the LiDAR post-processing (tools/rtH_lidar.sh, its training is skipped when done).
PYD=/root/gpufree-data/venvs/deploy/bin/python; PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
O=$H/runs/students_rtHv_s5/Z3; IT=2000
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_WAIST_LOCK=1
OA=$RS/out_rtHv; mkdir -p $OA
ALLB="--start-xy 0.05 --start-yaw 5 --dr-x --vel-est --head-bias 3 --height-bias 0.02"
for C in 1000 $IT; do
  while [ ! -f $O/student_$C.pt ]; do sleep 60; done
  sleep 30
  E=$H/deploy_exports/rtHv_s5_c$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=2 $PY deploy/export_student.py \
    --student $O/student_$C.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD tools/rtHv_meta.py $E/policy_meta.json
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "rtHv export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
  M=c$C; mkdir -p $OA/$M
  for L in $RS/v6/in/planned_layouts/*/; do
    L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
    [ $t = CrouchedTraverse ] && continue
    for c in RP10X RP10XS R3 RP10; do
      case $c in R3) X="--latency 0-20 --start rest --hold 3";;
        RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
        RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB";;
        RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02";; esac
      [ -f $OA/$M/${n%%_*}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $t --waist-lock \
        --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n%%_*}_$c.json > /dev/null 2>&1
    done
  done
  OC=$OA/cand_$M; mkdir -p $OC
  for CD in $H/runs/videos/e2e_robot_D/cand/*/*/; do
    CD=${CD%/}; lay=$(basename $(dirname $CD)); cl=$(basename $CD); t=$(echo $cl | cut -d_ -f1)
    [ $t = CrouchedTraverse ] && continue
    X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03"
    [ -f $OC/${lay}_$cl.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 32 --workers 100 --tasks $t --waist-lock \
      --motion-dirs $CD --latent-dir $CD $X --out $OC/${lay}_$cl.json > /dev/null 2>&1
  done
  $PYD tools/rtHv_summary.py $OA $IT > $OA/summary.md
  log "rtHv acceptance $M done"
done
log "rtHv done"
# LiDAR version: export + acceptance (training already finished or still running -> wait)
while [ ! -f $H/runs/students_rtHl_s5/Z3/student_2000.pt ]; do sleep 60; done
sleep 30
GPU=2 bash tools/rtH_lidar.sh
