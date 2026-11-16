#!/bin/bash
# (user): Deployment_H_mocap = bridge model WITHOUT the body velocity (rtGm c1000, Deployment_G_mocap)
# + stakes model WITH the body velocity estimated from the motion capture (as in Deployment_F).  This script trains the
# stakes model: student rtHv = DAgger from the Deployment_F student (rtFs c1500, trained with HNU_VEL_EST),
# teachers rtGw / rtGns (waist locked), waist roll / pitch locked, HNU_VEL_EST=1 (velocity like on the robot:
# mocap position slope over 60 ms, 10-40 ms delay, noisy, with the per-episode mocap bias), everything else as rtGm.
# Then export c1000 / c2000 (+parity), acceptance: 9 layouts x R3 / RP10 / RP10X / RP10XS (64 trials, --vel-est in the
# X conditions) and every stakes retrieval candidate (32 trials).  Usage: GPU=1 bash tools/rtH_vel.sh
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
NEWT=$(cat $N/rtGw_teachers.txt)
SRC=$H/runs/students_rtFs_s5/Z3/student_1500.pt
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_WAIST_LOCK=1
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=5
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_REST_RP=3 HNU_EP_LEN=25
      HNU_PERC_BIAS=0.04,0.04,5,0.3 HNU_VEL_EST=1"
O=${OUT:-$H/runs/students_rtHv_s5/Z3}
IT=${IT:-2000}
DIST="--num-envs ${NENV:-4096} --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30010 --scene geom
      --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC
      --latent-dir $D/z_tables_cv_f1_xzmix/Z3"
log "rtHv (mocap + velocity) student start from $SRC (out $O, $IT its)"
[ -f $O/student_$IT.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtHv CUDA_VISIBLE_DEVICES=${GPU:-1} $PY distill_train.py --profile rtHv_Z3 $NEWT \
  --iters $IT $DIST --out $O > $N/rtHv_student.log 2>&1
log "rtHv student: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
[ -n "$SMOKE" ] && exit 0
for C in 1000 $IT; do
  E=$H/deploy_exports/rtHv_s5_c$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=${GPU:-1} $PY deploy/export_student.py \
    --student $O/student_$C.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD tools/rtHv_meta.py $E/policy_meta.json
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "rtHv export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
done
# ---- acceptance (CPU)
OA=$RS/out_rtHv; mkdir -p $OA
ALLB="--start-xy 0.05 --start-yaw 5 --dr-x --vel-est --head-bias 3 --height-bias 0.02"
for M in c$IT c1000; do
  E=$H/deploy_exports/rtHv_s5_$M/L3; mkdir -p $OA/$M
  for L in $RS/v6/in/planned_layouts/*/; do
    L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
    for c in R3 RP10 RP10X RP10XS; do
      case $c in R3) X="--latency 0-20 --start rest --hold 3";;
        RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
        RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB";;
        RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02";; esac
      [ -f $OA/$M/${n%%_*}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $t --waist-lock \
        --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n%%_*}_$c.json > /dev/null 2>&1
    done
  done
  log "rtHv acceptance $M layouts done"
  # stakes retrieval candidates (chain tables), closest-to-real condition incl. stake placement errors
  OC=$OA/cand_$M; mkdir -p $OC
  for CD in $H/runs/videos/e2e_robot_D/cand/*/*/; do
    CD=${CD%/}; lay=$(basename $(dirname $CD)); cl=$(basename $CD); t=$(echo $cl | cut -d_ -f1)
    [ $t = CrouchedTraverse ] && continue
    X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03"
    [ -f $OC/${lay}_$cl.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 32 --workers 100 --tasks $t --waist-lock \
      --motion-dirs $CD --latent-dir $CD $X --out $OC/${lay}_$cl.json > /dev/null 2>&1
  done
  log "rtHv candidates $M done"
done
$PYD tools/rtHv_summary.py $OA $IT > $OA/summary.md
log "rtHv done"
