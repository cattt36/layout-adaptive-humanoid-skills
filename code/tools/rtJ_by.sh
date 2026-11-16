#!/bin/bash
# DEMO ONLY (not for the paper): second round of the stepping-on-stakes trackers.  5PR 016 had no usable
# stakes model (rtJbx_016: REP 31 %, falls 22/64, stakes 4-5 landing r95 0.5-0.75 m) -> three 016 variants, plus a
# continuation of 018 / 014 on the free GPUs.  Same settings as tools/rtJ_bx.sh (NEW + STK), 4000 more its:
#   rtJby_016a  resume rtJbx_016                                    (more training)
#   rtJby_016f  from the flat tracker rtJbxf_016 (falls less: 84 % no fall), stakes switched on
#   rtJby_016b  resume rtJbx_016, stake jitter 6 -> 3 cm, pushes x1.25 -> x1.0 (precision over robustness)
#   rtJby_018a  resume rtJbx_018;  rtJby_014a  resume rtJbx_014
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJby.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-4000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.06"
mkdir -p $S/lists
train() {  # task clip run source-run gpu env...
  local T=$1 C=$2 RUN=$3 SR=$4 G=$5; shift 5
  local SRC=$(last $T $SR) A=$S/lists/bm_${T}_$C.txt
  echo ${T}_$C > $A
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  log "$T $RUN from $SRC (gpu $G)"
  ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Stakes \
      --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(last $T $RUN)"
}
R5=HNU_STAKES_ROOT=$S/ref_stakes_s83_h10; R3=HNU_STAKES_ROOT=$S/ref_stakes_s93_h10
log "rtJby start ($IT its)"
train FiveStepPoleRun 016 rtJby_016a rtJbx_016 0 $NEW $STK $R5 & sleep 20
train FiveStepPoleRun 016 rtJby_016f rtJbxf_016 1 $NEW $STK $R5 & sleep 20
train FiveStepPoleRun 016 rtJby_016b rtJbx_016 2 $NEW $STK $R5 HNU_STAKE_JITTER=0.03 HNU_PUSH_SCALE=1.0 & sleep 20
train FiveStepPoleRun 018 rtJby_018a rtJbx_018 3 $NEW $STK $R5 & sleep 20
train ThreeStepPoleRun 014 rtJby_014a rtJbx_014 4 $NEW $STK $R3 &
wait
log "by train done"
