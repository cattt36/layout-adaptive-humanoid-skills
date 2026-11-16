#!/bin/bash
# DEPLOYMENT ONLY: STANDARD START for TODAY'S G1 (faulty waist) -- variant of tools/rtJ_std.sh (new G1).
# Waist roll / pitch locked around a random per-episode offset (HNU_WAIST_OFF: roll -0.20..+0.45, pitch -0.05..+0.50 rad,
# 25 % at 0; logs: pitch +0.06..+0.49, roll -0.13..+0.44 while standing), the policy reads the actual angle.
#   teachers rtJwt: from rtGw (stakes) / rtGns (bridge), waist locked, data/std, pendulum rope, 3000 its
#   students rtJwb (bridge, no velocity, from rtHwb c1500) / rtJws (stakes, mocap velocity, from rtHws c1500)
#   export + parity, acceptance (9 standard-start layouts) with the waist stuck at 0 / pitch +0.30 / roll +0.30 + pitch +0.40
# Waits for the teachers of rtJ_std.sh (GPUs 1-3) and uses GPUs 3 / 4.
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
TAGT=${TAGT:-rtJwt}; IT_T=${IT_T:-3000}; IT_S=${IT_S:-2000}
WOFF="HNU_WAIST_LOCK=1 HNU_WAIST_OFF=-0.2,0.45,-0.05,0.5,0.25"
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="$WOFF HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=10
     HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR_X_HIPKNEE=1 HNU_DR=1 HNU_DR_X=1 HNU_DR_LEVEL=2
     HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0
     HNU_REST_RP=3 HNU_EP_LEN=25"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_PUSH_SCALE=1.0 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05
     HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1"
W1="HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_PUSH_SCALE=1.0 HNU_CLIPS_ALLOW=$S/lists/teacher_w1_allow.txt
    HNU_STAKES_ROOT=$S/ref_stakes_s93_h10"
W2="HNU_STAKE_JITTER=0.08 HNU_PUSH_SCALE=1.25 HNU_STAKES_ROOT=$S/ref_stakes_s83_h10"
teacher() {  # task suffix gpu source-run env...
  local T=$1 SUF=$2 G=$3 SR=$4; shift 4
  local SRC=$(last $T $SR)
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$TAGT/model_*.pt 2>/dev/null)" ] && { log "$T $TAGT exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF --env.scene.num-envs ${NENV:-4096} \
      --agent.experiment-name hnu_track_$T --agent.run-name $TAGT --agent.max-iterations $IT_T --agent.resume True \
      --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$TAGT.log 2>&1 < /dev/null )
  log "$T $TAGT finished: $(last $T $TAGT)"
}
while ! grep -q "rtJ teachers rtJt done" $N/rtF.log; do sleep 60; done
log "rtJw teachers $TAGT start (from rtGw / rtGns, waist locked around random offsets, $IT_T its)"
teacher CrouchedTraverse -Bridge ${G_CT:-3} rtGns $NEW $BASE $BRG HNU_BASE_MID="$MID" HNU_BASE_END="$END" &
sleep 20
teacher ThreeStepPoleRun -Stakes ${G_W1:-4} rtGw $NEW $STK $W1 &
sleep 20
teacher FiveStepPoleRun -Stakes ${G_W2:-4} rtGw $NEW $STK $W2 &
wait
log "rtJw teachers $TAGT done"
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun $TAGT) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun $TAGT) --ckpt-CrouchedTraverse $(last CrouchedTraverse $TAGT)"
echo "$NEWT" > $N/${TAGT}_teachers.txt
SENV="$WOFF HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1
      HNU_STEP_STATES=$D/step_states/step_states_realF.npz $BASE
      HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
      HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
      HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
      HNU_GEOM_W1=$S/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$S/aug_xzmix_f1/ref_stakes_s83_h10
      HNU_BRIDGE_ROOT=$S/aug_xzmix_f1/ref_bridge_video_lift
      HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=10
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25
      HNU_PERC_BIAS=0.04,0.04,5,0.3 HNU_CLIPS_ALLOW=$S/lists/clips_train_xzmix.txt"
student() {  # tag gpu init extra-env...
  local TAG=$1 G=$2 SRC=$3; shift 3
  local O=$H/runs/students_${TAG}_s5/Z3
  [ -f $O/student_$IT_S.pt ] || env $SENV "$@" HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$TAG CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${TAG}_Z3 $NEWT --num-envs ${NENV:-4096} --iters $IT_S --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed 30013 --scene geom --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 \
    --init-from $SRC --latent-dir $S/z_tables_cv_f1_xzmix/Z3 --out $O > $N/${TAG}_student.log 2>&1
  log "$TAG student: $([ -f $O/student_$IT_S.pt ] && echo ok || echo FAILED)"
}
log "rtJw students start ($IT_S its)"
student rtJwb ${G_B:-3} $H/runs/students_rtHwb_s5/Z3/student_1500.pt HNU_LINVEL_ZERO=1 &
sleep 30
student rtJws ${G_S:-4} $H/runs/students_rtHws_s5/Z3/student_1500.pt HNU_VEL_EST=1 &
wait
# ---- export
for TAG in rtJwb rtJws; do
  for C in 1000 $IT_S; do
    E=$H/deploy_exports/${TAG}_c$C/L3; mkdir -p $E
    LZ=""; [ $TAG = rtJwb ] && LZ="HNU_LINVEL_ZERO=1"
    [ -f $E/policy.onnx ] || env $LZ HNU_WAIST_LOCK=1 HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift HNU_GEOM_W1=$S/ref_stakes_s93_h10 \
      HNU_GEOM_W2=$S/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
      HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 CUDA_VISIBLE_DEVICES=${G_B:-3} $PY deploy/export_student.py \
      --student $H/runs/students_${TAG}_s5/Z3/student_$C.pt --latent-dir $S/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
    LV=mocap; [ $TAG = rtJwb ] && LV=zero
    $PYD tools/rtJ_meta.py $E/policy_meta.json $LV waist
    [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
    log "$TAG export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
  done
done
# ---- acceptance (CPU): waist stuck at 0 / pitch +0.30 / roll +0.30 & pitch +0.40
OA=$RS/out_rtJw; mkdir -p $OA
export HNU_S2S_ROPE_LEN=2.0 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID="$MID" HNU_BASE_END="$END"
for TAG in rtJwb rtJws; do
  for C in $IT_S 1000; do
    M=${TAG}_c$C; E=$H/deploy_exports/$M/L3; mkdir -p $OA/$M
    for L in $RS/std/planned_layouts/*/; do
      L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
      [ $TAG = rtJwb ] && [ $t != CrouchedTraverse ] && continue
      [ $TAG = rtJws ] && [ $t = CrouchedTraverse ] && continue
      VE=""; [ $TAG = rtJws ] && VE="--vel-est"
      ALLB="--start-xy 0.05 --start-yaw 10 --dr-x $VE --head-bias 3 --height-bias 0.02"
      SJ=""; [ $t != CrouchedTraverse ] && SJ="--stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"
      X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB $SJ"
      for w in 0,0 0,0.30 0.30,0.40; do
        tag=W$(echo $w | tr ',' '_')
        [ -f $OA/$M/${n%%_*}_$tag.json ] || HNU_S2S_WAIST_OFF=$w $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 \
          --tasks $t --waist-lock --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n%%_*}_$tag.json > /dev/null 2>&1
      done
    done
    log "rtJw acceptance $M done"
  done
done
$PYD tools/rtJ_summary.py $OA $IT_S waist > $OA/summary.md
log "rtJw done"
