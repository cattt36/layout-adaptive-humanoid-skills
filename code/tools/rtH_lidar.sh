#!/bin/bash
# LiDAR version (backup of the mocap version, user): obstacle features from a LiDAR-odometry pose of the
# pelvis in a prior map of the venue instead of the motion capture.  Student rtHl = DAgger from the mocap student
# rtGm student_1000 (Deployment_G_mocap), same teachers / randomisation / waist lock / no velocity input, plus the
# LiDAR pose model HNU_POSE_RATE (hnu_tasks/perception.py _lidar_pose): per episode 60 % SCAN mode (10 Hz updates =
# every 5 control steps, delay 40-100 ms) or 40 % FAST mode (IMU-propagated pose, every step, delay 20-60 ms), noise
# 1 cm / 0.5 deg per update; the per-episode pose bias of the mocap version (4 cm / 5 deg) stands for the map
# registration error.  sim2sim before (mocap student, 64 trials): 10 Hz + 40 ms -> D1/D2/N3 72-81 %, stakes 2-11 %.
# Then: export c1000 / c2000 + parity, acceptance (9 layouts x MOCAP / L1 / P50 / P10a / P10b / P10aS).
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
NEWT=$(cat $N/rtGw_teachers.txt)
SRC=$H/runs/students_rtGm_s5/Z3/student_1000.pt
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_WAIST_LOCK=1 HNU_LINVEL_ZERO=1
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=5
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_REST_RP=3 HNU_EP_LEN=25
      HNU_PERC_BIAS=0.04,0.04,5,0.3 HNU_POSE_RATE=5,2,5,0.4,1,3,0.01,0.5"
O=${OUT:-$H/runs/students_rtHl_s5/Z3}
IT=${IT:-2000}
DIST="--num-envs ${NENV:-4096} --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30009 --scene geom
      --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC
      --latent-dir $D/z_tables_cv_f1_xzmix/Z3"
log "rtH lidar student start from $SRC (out $O, $IT its)"
[ -f $O/student_$IT.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtHl CUDA_VISIBLE_DEVICES=${GPU:-0} $PY distill_train.py --profile rtHl_Z3 $NEWT \
  --iters $IT $DIST --out $O > $N/rtHl_student.log 2>&1
log "rtH lidar student: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
[ -n "$SMOKE" ] && exit 0
for C in 1000 $IT; do
  E=$H/deploy_exports/rtHl_s5_c$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=${GPU:-0} $PY deploy/export_student.py \
    --student $O/student_$C.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD tools/rtH_meta.py $E/policy_meta.json
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "rtH export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
done
# ---- acceptance (CPU): new student c1000 / c2000 vs the mocap student under the LiDAR conditions
OA=$RS/out_rtHl; mkdir -p $OA
B="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --dr-x --start-xy 0.05 --start-yaw 5"
for M in rtHl_s5_c$IT rtHl_s5_c1000 rtGm_s5_c1000; do
  E=$H/deploy_exports/$M/L3; mkdir -p $OA/$M
  for L in $RS/v6/in/planned_layouts/*/; do
    L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2); SJ=""
    [ $t != CrouchedTraverse ] && SJ="--stake-jitter 0.03 --obst-shift 0.03"
    for c in MOCAP L1 P50 P10a P10b P10aS; do
      HZ=0; LG=0
      case $c in
        MOCAP) X="$B --head-bias 3 --height-bias 0.02";;
        L1) X="$B --init-off 0.02 --drift 0.01";;
        P50) X="$B --init-off 0.02 --drift 0.01"; HZ=50; LG=0.03;;
        P10a) X="$B --init-off 0.02 --drift 0.01"; HZ=10; LG=0.05;;
        P10b) X="$B --init-off 0.02 --drift 0.01"; HZ=10; LG=0.10;;
        P10aS) X="$B --init-off 0.02 --drift 0.01 $SJ --perc-noise 0.02"; HZ=10; LG=0.05;;
      esac
      # mocap student: L1 / P50 / P10a / P10b already in out_rtGm/lio
      [ $M = rtGm_s5_c1000 ] && [ -f $RS/out_rtGm/lio/${n%%_*}_$c.json ] && continue
      [ -f $OA/$M/${n%%_*}_$c.json ] || HNU_S2S_POSE_HZ=$HZ HNU_S2S_POSE_LAG_S=$LG $PYD deploy/sim2sim.py $E --crit task \
        --trials 64 --workers 100 --tasks $t --waist-lock --motion-dirs $L --latent-dir $L $X \
        --out $OA/$M/${n%%_*}_$c.json > /dev/null 2>&1
    done
  done
  log "rtH acceptance $M done"
done
$PYD tools/rtH_summary.py $OA $RS/out_rtGm/lio $IT > $OA/summary.md
log "rtH done"
