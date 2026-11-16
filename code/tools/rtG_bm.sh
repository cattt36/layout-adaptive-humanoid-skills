#!/bin/bash
# (user): BeyondMimic-style BLIND trackers for the real robot, one per task: the teachers (mjlab tracking,
# PPO) fine-tuned with HNU_ACTOR_BLIND=1 (actor: reference command, anchor orientation, base angular velocity, joints,
# last action -- no anchor position, no base linear velocity, no obstacle perception; critic privileged), waist roll /
# pitch locked, rope start (0.9 BW released over 1-12 s), real stepping states, contact / actuator randomisation,
# start placement +-5 cm / +-5 deg (unobservable), no swing-scuff penalty (user: allow swing-foot touches).
# From the round-3 teachers (stakes rtGw, CT rtGns).  IT its (default 3000).  Usage: IT=3000 bash tools/rtG_bm.sh
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 PYTHONPATH=$H/code
NEW="HNU_ACTOR_BLIND=1 HNU_WAIST_LOCK=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.05 HNU_REST_YAW=5 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_REST_RP=3 HNU_EP_LEN=25"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_PUSH_SCALE=1.0 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05
     HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1"
W1="HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_PUSH_SCALE=1.0
    HNU_CLIPS_ALLOW=$D/clip_split_trial/teacher_w1_allow.txt HNU_STAKES_ROOT=$D/ref_stakes_s93_h10"
W2="HNU_STAKE_JITTER=0.08 HNU_PUSH_SCALE=1.25 HNU_STAKES_ROOT=$D/ref_stakes_s83_h10"
RUN=${RUN:-rtGbm}; IT=${IT:-3000}
train() {  # task suffix gpu src iters env...
  local T=$1 SUF=$2 G=$3 SRC=$4; shift 4
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
# continuation: RUN=rtGbm2 SRCRUN=rtGbm IT=4000 bash tools/rtG_bm.sh (all three tasks from SRCRUN)
SC=${SRCRUN:-rtGns}; SS=${SRCRUN:-rtGw}
log "blind trackers $RUN start (from $SC / $SS)"
train CrouchedTraverse -Bridge ${G_CT:-1} $(last CrouchedTraverse $SC) $NEW $BASE $BRG HNU_BASE_MID="$MID" HNU_BASE_END="$END" &
sleep 20
train ThreeStepPoleRun -Stakes ${G_W1:-2} $(last ThreeStepPoleRun $SS) $NEW $STK $W1 &
sleep 20
train FiveStepPoleRun -Stakes ${G_W2:-3} $(last FiveStepPoleRun $SS) $NEW $STK $W2 &
wait
log "blind trackers $RUN done"
