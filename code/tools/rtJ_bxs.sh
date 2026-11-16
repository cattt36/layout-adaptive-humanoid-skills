#!/bin/bash
# DEMO ONLY (user: lab mocap too inaccurate; leadership demo = the robot replays the three skills, props placed
# where the robot will not touch them; the real experiment is redone later at another mocap venue; NOT for the paper).
# BeyondMimic-style pure trackers trained by RL (PPO, no distillation), ONE CLIP per policy, NO mocap in training inputs or
# at run time: the actor sees reference command, anchor orientation (IMU), base angular velocity, joints, last action
# (HNU_ACTOR_BLIND=1: anchor position, base linear velocity, obstacle perception = 0); critic privileged.
# Waist free (new G1), standard start (robot stands facing the obstacles, data/std), rope start (pendulum), pushes x1.25,
# contact / actuator / payload randomisation, 0-20 ms sensor lag, start placement +-2 cm / +-3 deg.
# Clips chosen for little turning (tools/clip_turns.py; user face the obstacles, avoid turning):
#   CT 005 / 013 (bridge, posts +-10 cm, bridge +-5 cm, height -8/+6 cm), 3PR 014 / 012, 5PR 018 / 016:
#   rtJbx_<nnn>  = stepping ON the stakes (stakes +-6 cm)          [HNU-Track-*-Stakes, data/std]
#   rtJbxf_<nnn> = flat floor (10 cm raise removed, data/std_flat) [stakes hidden, stake objectives off]
# vs the first attempt rtJbm (stopped at ~500 its): the blind policy cannot see its global position, so the horizontal
# drift terminations (0.4 / 0.5 m) ended almost every episode -> drift limit 1.0 m, global root reward std 0.2 -> 0.4.
# From the standard-start teachers rtJt, 4000 its.  Log runs/night/rtF.log, last line "bx train done".
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
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05
     HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.06"
FLAT="HNU_PERCEPTION=1 HNU_RUNUP_FREE_S=-1 HNU_STAKE_RW=0 HNU_STAKE_CONTACT=0 HNU_STAKE_MISS_TERM=0
      HNU_STAKE_JITTER=0 HNU_STAKE_STRAIGHT=0 HNU_SLIP_W=-0.1 HNU_FEET_STD=0.2 HNU_STAKES_ROOT=$D/std_flat"
mkdir -p $S/lists
train() {  # task clip run-prefix gpu env...
  local T=$1 C=$2 P=$3 G=$4; shift 4
  local RUN=$P$C SRC=$(last $T rtJt) A=$S/lists/bm_${T}_$C.txt
  echo ${T}_$C > $A
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  local SUF=-Stakes; [ $T = CrouchedTraverse ] && SUF=-Bridge
  ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
# ---- self-calibrated stakes round (user OK): the training stakes are moved to the stakes-on tracker's own
# mean landings (robot_sim_check/bmJ/calib/on_<clip>/<clip>_stakes.json, tools/calib_stakes.py) = what the real-robot
# procedure does (stakes moved to the measured landings), stake jitter +-6 cm kept; fine-tuned from rtJbx_<clip>.
# Usage: ROUND=1 SRCP=rtJbx_ CAL=<dir with on_<clip>/<clip>_stakes.json> bash tools/rtJ_bxs.sh
ROUND=${ROUND:-1}; SRCP=${SRCP:-rtJbx_}; CAL=${CAL:-/root/gpufree-data/robot_sim_check/bmJ/calib}
X=rtJbs${ROUND}; IT=${IT_S:-2000}; DC=$D/std_cal$ROUND
for cs in ThreeStepPoleRun_012:s93 ThreeStepPoleRun_014:s93 FiveStepPoleRun_018:s83; do
  c=${cs%%:*}; T=${c%_*}; R=ref_stakes_${cs##*:}_h10; mkdir -p $DC/$R/$T
  cp $S/$R/$T/${c}_motion.npz $DC/$R/$T/; cp $CAL/on_$c/${c}_stakes.json $DC/$R/$T/
done
trainc() {  # task clip gpu root
  local T=$1 C=$2 G=$3 RT=$4 RUN=${X}_$2
  local SRC=$(last $T ${SRCP}$C) A=$S/lists/bm_${T}_$C.txt
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  ( cd $H/runs && env $NEW $STK HNU_STAKES_ROOT=$RT HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
log "$X self-calibrated stakes trackers start ($IT its, stakes from $CAL, from ${SRCP}<clip>)"
trainc ThreeStepPoleRun 012 ${G1:-3} $DC/ref_stakes_s93_h10 & sleep 20
trainc ThreeStepPoleRun 014 ${G2:-4} $DC/ref_stakes_s93_h10 & sleep 20
trainc FiveStepPoleRun 018 ${G3:-0} $DC/ref_stakes_s83_h10 &
wait
log "bs$ROUND train done"
