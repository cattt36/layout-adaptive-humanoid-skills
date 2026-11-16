#!/bin/bash
# DEMO faulty waist, round 3 for the stakes: continue the fixed-stake trackers of tools/rtJ_bv.sh (rtJbv_<tag>)
# for 3000 more its (3PR 012 82-92 %, 014 41-93 %, 5PR 018A 34-56 % after round 2):
#   rtJbv2_012 / 014 / 018A  : same env, continued                                  gpus 0 / 1 / 2
#   rtJbv2_018Aj0            : 5PR human pattern, stakes exact (no placement jitter)   gpu 3
#   rtJbv2_018Ap             : 5PR human pattern, pushes x1.0 (x1.25 before)           gpu 4
# Eval: RP=rtJbv2 OD=bv2 tools/rtJ_bv_eval.sh -> robot_sim_check/bmJ/bv2.  Log runs/night/rtJbv2.log.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbv2.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-3000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
WOFF="HNU_WAIST_LOCK=1 HNU_WAIST_OFF=-0.2,0.45,-0.05,0.5,0.25"
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.02"
train() {  # task clip tag source-tag gpu env...
  local T=$1 C=$2 TAG=$3 ST=$4 G=$5; shift 5
  local RUN=rtJbv2_$TAG SRC=$(last $T rtJbv_$ST) A=$S/lists/bw_${T}_$C.txt
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env "$@" $WOFF HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  RP=rtJbv2 OD=bv2 G=$G bash tools/rtJ_bv_eval.sh $T $C $TAG >> $N/rtJbv2_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbv2 start"
train ThreeStepPoleRun 012 012 012 0 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train ThreeStepPoleRun 014 014 014 1 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train FiveStepPoleRun 018 018A 018A 2 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 & sleep 20
train FiveStepPoleRun 018 018Aj0 018A 3 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 HNU_STAKE_JITTER=0 & sleep 20
train FiveStepPoleRun 018 018Ap 018A 4 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 HNU_PUSH_SCALE=1.0 &
wait
OD=bv2 TAGS="012 014 018A 018Aj0 018Ap" bash tools/rtJ_bv_eval.sh summary > $B/bv2/summary.md 2>&1
log "rtJbv2 done"
