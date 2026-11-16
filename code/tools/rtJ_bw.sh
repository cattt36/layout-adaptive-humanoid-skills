#!/bin/bash
# DEMO ONLY (not for the paper).  User: the same blind demo trackers for the OLD G1 (faulty waist motor).
# = the shipped Deployment_K_demo trackers fine-tuned with the waist roll / pitch LOCKED around a random per-episode
# offset (as tools/rtJw.sh for the faulty-waist students: HNU_WAIST_OFF roll -0.20..+0.45, pitch -0.05..+0.50 rad,
# 25 % at 0; logs: pitch +0.06..+0.49, roll -0.13..+0.44 while standing); waist action scale 0 -> the
# robot gets the constant waist target 0, the policy reads the actual waist angle.  Same tasks / props as vK:
#   rtJbw_013 from rtJbz_013w10 (bridge at the reference pose, +-2 cm, contact weight -10)   gpu 0
#   rtJbw_005 from rtJbz_005j2  (backup bridge)                                             gpu 1
#   rtJbw_012 / 014 from rtJbx_012 / 014 (3PR on stakes), rtJbw_018 from rtJbx_018   gpus 2 / 3 / 4
# Each run is exported + evaluated as soon as it finishes (tools/rtJ_bw_eval.sh).  Log runs/night/rtJbw.log.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbw.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-3000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
WOFF="HNU_WAIST_LOCK=1 HNU_WAIST_OFF=-0.2,0.45,-0.05,0.5,0.25"
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_SLIP_W=-0.3 HNU_BRIDGE_ROOT=$D/std_ctreal
     HNU_BRIDGE_JITTER_XY=0.02 HNU_BRIDGE_JITTER_ZLO=-0.01 HNU_BRIDGE_JITTER_ZHI=0.01 HNU_BRIDGE_POST_JITTER=0.02"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.06"
train() {  # task clip source-run gpu env...
  local T=$1 C=$2 SR=$3 G=$4; shift 4
  local RUN=rtJbw_$C SRC=$(last $T $SR) A=$S/lists/bw_${T}_$C.txt
  echo ${T}_$C > $A
  local SUF=-Stakes; [ $T = CrouchedTraverse ] && SUF=-Bridge
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env "$@" $WOFF HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T$SUF --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  G=$G bash tools/rtJ_bw_eval.sh $T $C >> $N/rtJbw_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbw start"
train CrouchedTraverse 013 rtJbz_013w10 0 $NEW $BASE $BRG HNU_BRIDGE_W=-10.0 & sleep 20
train CrouchedTraverse 005 rtJbz_005j2 1 $NEW $BASE $BRG & sleep 20
train ThreeStepPoleRun 012 rtJbx_012 2 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train ThreeStepPoleRun 014 rtJbx_014 3 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train FiveStepPoleRun 018 rtJbx_018 4 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 &
wait
bash tools/rtJ_bw_eval.sh summary > /root/gpufree-data/robot_sim_check/bmJ/bw/summary.md 2>&1
log "rtJbw done"
