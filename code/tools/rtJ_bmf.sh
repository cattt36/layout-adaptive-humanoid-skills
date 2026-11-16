#!/bin/bash
# DEMO ONLY: FLAT-GROUND variant of the stakes demo trackers (tools/rtJ_bm.sh): the stepping-stone motion
# on the floor, props beside the footholds (user: both variants).  Reference = the standard-start stakes clip with the
# 10 cm raise removed (tools/flatten_stakes.py -> data/std_flat, no stakes json -> stakes hidden, stake objectives off).
# Same blind actor / rope start / randomisation as rtJ_bm.sh; global root / feet tracking kept on for a repeatable
# path (HNU_RUNUP_FREE_S=-1: never gated).  From the standard-start stakes teachers rtJt.
# Out: runs/logs/rsl_rl/hnu_track_<task>/*_rtJbmf_<nnn>/   log runs/night/rtF.log ("rtJbmf train done")
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
FLAT="HNU_PERCEPTION=1 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=-1 HNU_STAKE_RW=0 HNU_STAKE_CONTACT=0 HNU_STAKE_MISS_TERM=0
      HNU_STAKE_JITTER=0 HNU_STAKE_STRAIGHT=0 HNU_SLIP_W=-0.1 HNU_STAKES_ROOT=$D/std_flat"
mkdir -p $S/lists
train() {  # task clip gpu env...
  local T=$1 C=$2 G=$3; shift 3
  local RUN=${RUNP:-rtJbmf}_$C SRC=$(last $T rtJt) A=$S/lists/bm_${T}_$C.txt
  echo ${T}_$C > $A
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
log "rtJbmf flat demo trackers start ($IT its)"
train ThreeStepPoleRun 014 ${G_S1:-1} $NEW $FLAT &
sleep 20
train ThreeStepPoleRun 012 ${G_S2:-2} $NEW $FLAT &
sleep 20
train FiveStepPoleRun 018 ${G_F1:-3} $NEW $FLAT &
sleep 20
train FiveStepPoleRun 020 ${G_F2:-0} $NEW $FLAT &
wait
log "rtJbmf train done"
