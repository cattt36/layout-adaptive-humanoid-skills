#!/bin/bash
# /05 DEPLOYMENT ONLY (paper models / data untouched): STANDARD START for the new G1 (healthy waist -> not
# locked).  Every clip starts with the robot standing at the layout origin in a normal stance facing +x; the reference
# itself does the step-in (one step) + slow-in into the skill (tools/make_start.py, data/std = make_start_batch.py).
#   teachers rtJt: from rtFd (waist free), 3000 its, data/std roots, rope with a pendulum pull (HNU_ROPE_LEN 2 m),
#                     start placement +-5 cm / +-10 deg, no stepping starts
#   students: rtJb (bridge model, no body velocity) and rtJs (stakes model, mocap velocity), DAgger from the
#             Deployment_F student (rtFs c1500, waist free), 2000 its
#   export c1000 / c2000 + parity, acceptance on the 9 standard-start layouts (robot_sim_check/std/planned_layouts)
# Usage: bash tools/rtJ_std.sh   (SMOKE=1 IT_T=3 IT_S=3 NENV=512 for a smoke test)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
TAGT=${TAGT:-rtJt}; IT_T=${IT_T:-3000}; IT_S=${IT_S:-2000}
# allow lists restricted to the standard-start clips
mkdir -p $S/lists
for c in $(cat $D/clip_split_trial/teacher_w1_allow.txt); do [ -f $S/ref_stakes_s93_h10/ThreeStepPoleRun/${c}_motion.npz ] && echo $c; done > $S/lists/teacher_w1_allow.txt
for c in $(cat $D/clip_split_cv/fold1/clips_train_xzmix.txt); do t=${c%%_*}; r=ref_bridge_video_lift; [ $t = ThreeStepPoleRun ] && r=ref_stakes_s93_h10; [ $t = FiveStepPoleRun ] && r=ref_stakes_s83_h10
  [ -f $S/aug_xzmix_f1/$r/$t/${c}_motion.npz ] && [ -f $S/z_tables_cv_f1_xzmix/Z3/${c}_z.npy ] && echo $c; done > $S/lists/clips_train_xzmix.txt
log "rtJ standard start: $(wc -l < $S/lists/teacher_w1_allow.txt) 3PR teacher clips, $(wc -l < $S/lists/clips_train_xzmix.txt) student clips"
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=10
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
teacher() {  # task suffix gpu env...
  local T=$1 SUF=$2 G=$3; shift 3
  local SRC=$(last $T rtFd)
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$TAGT/model_*.pt 2>/dev/null)" ] && [ -z "$SMOKE" ] && { log "$T $TAGT exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF --env.scene.num-envs ${NENV:-4096} \
      --agent.experiment-name hnu_track_$T --agent.run-name $TAGT --agent.max-iterations $IT_T --agent.resume True \
      --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$TAGT.log 2>&1 < /dev/null )
  log "$T $TAGT finished: $(last $T $TAGT)"
}
log "rtJ teachers $TAGT start (from rtFd, $IT_T its)"
teacher CrouchedTraverse -Bridge ${G_CT:-1} $NEW $BASE $BRG HNU_BASE_MID="$MID" HNU_BASE_END="$END" &
sleep 20
teacher ThreeStepPoleRun -Stakes ${G_W1:-2} $NEW $STK $W1 &
sleep 20
teacher FiveStepPoleRun -Stakes ${G_W2:-3} $NEW $STK $W2 &
wait
log "rtJ teachers $TAGT done"
[ -n "$SMOKE_T" ] && exit 0
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun $TAGT) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun $TAGT) --ckpt-CrouchedTraverse $(last CrouchedTraverse $TAGT)"
echo "$NEWT" > $N/${TAGT}_teachers.txt
SRC=$H/runs/students_rtFs_s5/Z3/student_1500.pt
SENV="HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1
      HNU_STEP_STATES=$D/step_states/step_states_realF.npz $BASE
      HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
      HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
      HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
      HNU_GEOM_W1=$S/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$S/aug_xzmix_f1/ref_stakes_s83_h10
      HNU_BRIDGE_ROOT=$S/aug_xzmix_f1/ref_bridge_video_lift
      HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=10
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25
      HNU_PERC_BIAS=0.04,0.04,5,0.3 HNU_CLIPS_ALLOW=$S/lists/clips_train_xzmix.txt"
student() {  # tag gpu extra-env...
  local TAG=$1 G=$2; shift 2
  local O=$H/runs/students_${TAG}_s5/Z3
  [ -f $O/student_$IT_S.pt ] || env $SENV "$@" HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$TAG CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${TAG}_Z3 $NEWT --num-envs ${NENV:-4096} --iters $IT_S --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed 30012 --scene geom --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 \
    --init-from $SRC --latent-dir $S/z_tables_cv_f1_xzmix/Z3 --out $O > $N/${TAG}_student.log 2>&1
  log "$TAG student: $([ -f $O/student_$IT_S.pt ] && echo ok || echo FAILED)"
}
log "rtJ students start ($IT_S its)"
student rtJb ${G_B:-1} HNU_LINVEL_ZERO=1 &
sleep 30
student rtJs ${G_S:-2} HNU_VEL_EST=1 &
wait
[ -n "$SMOKE" ] && exit 0
# ---- export
for TAG in rtJb rtJs; do
  for C in 1000 $IT_S; do
    E=$H/deploy_exports/${TAG}_c$C/L3; mkdir -p $E
    LZ=""; [ $TAG = rtJb ] && LZ="HNU_LINVEL_ZERO=1"
    [ -f $E/policy.onnx ] || env $LZ HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift HNU_GEOM_W1=$S/ref_stakes_s93_h10 HNU_GEOM_W2=$S/ref_stakes_s83_h10 \
      HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 \
      CUDA_VISIBLE_DEVICES=${G_B:-1} $PY deploy/export_student.py --student $H/runs/students_${TAG}_s5/Z3/student_$C.pt \
      --latent-dir $S/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
    [ $TAG = rtJb ] && $PYD tools/rtJ_meta.py $E/policy_meta.json zero || $PYD tools/rtJ_meta.py $E/policy_meta.json mocap
    [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
    log "$TAG export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
  done
done
# ---- acceptance (CPU): 9 standard-start layouts, pendulum rope
OA=$RS/out_rtJ; mkdir -p $OA
export HNU_S2S_ROPE_LEN=2.0 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID="$MID" HNU_BASE_END="$END"
for TAG in rtJb rtJs; do
  for C in $IT_S 1000; do
    M=${TAG}_c$C; E=$H/deploy_exports/$M/L3; mkdir -p $OA/$M
    for L in $RS/std/planned_layouts/*/; do
      L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
      [ $TAG = rtJb ] && [ $t != CrouchedTraverse ] && continue
      [ $TAG = rtJs ] && [ $t = CrouchedTraverse ] && continue
      VE=""; [ $TAG = rtJs ] && VE="--vel-est"
      ALLB="--start-xy 0.05 --start-yaw 10 --dr-x $VE --head-bias 3 --height-bias 0.02"
      for c in R3 RP10 RP10X RP10XS; do
        case $c in R3) X="--latency 0-20 --start rest --hold 3";;
          RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
          RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB";;
          RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02";; esac
        [ -f $OA/$M/${n%%_*}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $t \
          --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n%%_*}_$c.json > /dev/null 2>&1
      done
    done
    log "rtJ acceptance $M done"
  done
done
$PYD tools/rtJ_summary.py $OA $IT_S > $OA/summary.md
log "rtJ done"
