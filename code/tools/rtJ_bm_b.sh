#!/bin/bash
# replacement runs (CT 028 -> 013, 5PR 020 -> 016: straighter clips); otherwise identical to the script below.
# DEMO ONLY (user: lab mocap too inaccurate; leadership demo = the robot replays the three skills and the
# props are placed along the robot's own path; real experiment redone later at another mocap venue; NOT for the paper).
# BeyondMimic-style pure tracker, ONE CLIP per policy, NO mocap at run time: the actor sees the reference command,
# anchor orientation (IMU), base angular velocity, joints, last action (HNU_ACTOR_BLIND=1: anchor position, base linear
# velocity, obstacle perception = 0); critic privileged.  Waist free (new G1), standard start (data/std), rope start,
# obstacle positions randomised around the reference path (bridge whole +-5 cm, posts +-10 cm, height -8/+6 cm;
# stakes +-6 cm), start placement +-2 cm / +-3 deg (floor foot template), pushes x1.25, contact / actuator / payload
# randomisation, 0-20 ms sensor lag.  Fine-tuned from the standard-start teachers rtJt.
# Out: runs/logs/rsl_rl/hnu_track_<task>/*_rtJbm_<nnn>/   log runs/night/rtF.log ("rtJbm train done")
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-4000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05
     HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.06"
mkdir -p $S/lists
train() {  # task clip gpu env...
  local T=$1 C=$2 G=$3; shift 3
  local RUN=${RUNP:-rtJbm}_$C SRC=$(last $T rtJt) A=$S/lists/bm_${T}_$C.txt
  echo ${T}_$C > $A
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  local SUF=-Stakes; [ $T = CrouchedTraverse ] && SUF=-Bridge
  ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
log "rtJbm_b demo trackers start ($IT its): CT 013, 5PR 016 (stakes on) -- straighter clips (user face the obstacles, avoid turning)"
train CrouchedTraverse 013 ${G_C2:-2} $NEW $BASE $BRG HNU_BASE_MID="$MID" HNU_BASE_END="$END" &
sleep 20
train FiveStepPoleRun 016 ${G_F2:-4} $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 &
wait
log "rtJbm_b train done"
