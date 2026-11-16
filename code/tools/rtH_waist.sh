#!/bin/bash
# backup for the robot with the faulty waist (user: train in advance): waist roll / pitch locked around a
# random per-episode offset (HNU_WAIST_OFF: roll -0.15..+0.10, pitch -0.05..+0.25 rad, 30 % at 0), the policy sees the
# actual waist angle in its joint observations.  sim2sim of the vH models with the waist stuck at pitch +0.20 rad:
# bridge 34-44 %, stakes 59-88 %.  Two students (DAgger, same teachers):
#   rtHwb (bridge model): from rtGm student_1000, no body velocity (as model_bridge)
#   rtHws (stakes model): from rtHv student_2000, mocap velocity (as model_stakes)
# Then export + parity, acceptance on the 9 layouts with the waist stuck at 0 / pitch +0.10 / +0.20 / roll -0.10 +
# pitch +0.10 / pitch -0.10 (RP10X, stakes + placement errors), same for the vH models (comparison).
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
NEWT=$(cat $N/rtGw_teachers.txt)
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_WAIST_LOCK=1
WOFF=${WOFF:--0.15,0.10,-0.05,0.25,0.3}
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=5
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_REST_RP=3 HNU_EP_LEN=25
      HNU_PERC_BIAS=0.04,0.04,5,0.3 HNU_WAIST_OFF=$WOFF"
IT=${IT:-1500}
student() {  # tag src gpu extra-env...
  local TAG=$1 SRC=$2 G=$3; shift 3
  local O=${OUTROOT:-$H/runs}/students_${TAG}_s5/Z3
  [ -f $O/student_$IT.pt ] || env $AUG $NEWS "$@" HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
    HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$TAG CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${TAG}_Z3 $NEWT \
    --num-envs ${NENV:-4096} --iters $IT --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30011 --scene geom \
    --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC \
    --latent-dir $D/z_tables_cv_f1_xzmix/Z3 --out $O > $N/${TAG}_student.log 2>&1
  log "$TAG student: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
}
log "rtH waist-offset students start ($WOFF, $IT its)"
student rtHwb $H/runs/students_rtGm_s5/Z3/student_1000.pt ${G_B:-2} HNU_LINVEL_ZERO=1 &
sleep 30
student rtHws $H/runs/students_rtHv_s5/Z3/student_2000.pt ${G_S:-3} HNU_VEL_EST=1 &
wait
[ -n "$SMOKE" ] && exit 0
for TAG in rtHwb rtHws; do
  E=$H/deploy_exports/${TAG}_s5_c$IT/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=${G_B:-2} $PY deploy/export_student.py \
    --student $H/runs/students_${TAG}_s5/Z3/student_$IT.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD tools/rtHv_meta.py $E/policy_meta.json
  [ $TAG = rtHwb ] && $PYD tools/rtHw_meta.py $E/policy_meta.json zero || $PYD tools/rtHw_meta.py $E/policy_meta.json mocap
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "$TAG export: $(tail -n 1 $E/parity.txt | cut -c1-60)"
done
# ---- acceptance (CPU): waist stuck at offsets, new vs vH models
O=$RS/out_rtHw; mkdir -p $O
B="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.05 --start-yaw 5 --dr-x --head-bias 3 --height-bias 0.02"
for L in $RS/v6/in/planned_layouts/*/; do
  L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
  if [ $t = CrouchedTraverse ]; then MS="rtHwb_s5_c$IT rtGm_s5_c1000"; X="$B"; else MS="rtHws_s5_c$IT rtHv_s5_c2000"; X="$B --vel-est --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"; fi
  for M in $MS; do
    mkdir -p $O/$M
    for w in "0,0" "0,0.10" "0,0.20" "-0.10,0.10" "0,-0.10"; do
      tag=$(echo $w | tr ',' '_')
      [ -f $O/$M/${n%%_*}_w$tag.json ] || HNU_S2S_WAIST_OFF=$w $PYD deploy/sim2sim.py $H/deploy_exports/$M/L3 --crit task \
        --trials 64 --workers 100 --tasks $t --waist-lock --motion-dirs $L --latent-dir $L $X --out $O/$M/${n%%_*}_w$tag.json > /dev/null 2>&1
    done
  done
done
$PYD tools/rtHw_summary.py $O $IT > $O/summary.md
log "rtH waist done"
