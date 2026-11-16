#!/bin/bash
# DEMO faulty waist, round 2 for the STAKES (tools/rtJ_bw.sh stake trackers land 20-40 cm off the vK stakes
# and spread a lot; one calibration round only gave 3PR 46-73 %).  Same recipe as the bridge (which worked): the props are
# FIXED where they will stand (+-2 cm placement error only) instead of +-6 cm random, so the blind tracker learns to hit them:
#   rtJbv_012 / 014 : human 3PR pattern (ref_stakes_s93_h10) exactly                         gpus 0 / 1
#   rtJbv_018A      : human 5PR pattern (ref_stakes_s83_h10)                                    gpu 2
#   rtJbv_018B      : 5PR option B (vK "compressed" layout, bmJ/stk_final/018B)            gpu 3
#   rtJbv_018Aw     : as 018A with stake reward weight 10 (5 before)                             gpu 4
# From rtJbw_<clip> (waist already locked around random offsets), 2500 its, then tools/rtJ_bv_eval.sh.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbv.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-2500}
mkdir -p $D/std_bvB/FiveStepPoleRun
cp $B/stk_final/018B/FiveStepPoleRun_018_motion.npz $B/stk_final/018B/FiveStepPoleRun_018_stakes.json $D/std_bvB/FiveStepPoleRun/
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
train() {  # task clip tag gpu env...
  local T=$1 C=$2 TAG=$3 G=$4; shift 4
  local RUN=rtJbv_$TAG SRC=$(last $T rtJbw_$C) A=$S/lists/bw_${T}_$C.txt
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env "$@" $WOFF HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  G=$G bash tools/rtJ_bv_eval.sh $T $C $TAG >> $N/rtJbv_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbv start"
train ThreeStepPoleRun 012 012 0 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train ThreeStepPoleRun 014 014 1 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s93_h10 & sleep 20
train FiveStepPoleRun 018 018A 2 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 & sleep 20
train FiveStepPoleRun 018 018B 3 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB & sleep 20
train FiveStepPoleRun 018 018Aw 4 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 HNU_STAKE_RW=10 &
wait
bash tools/rtJ_bv_eval.sh summary > $B/bv/summary.md 2>&1
log "rtJbv done"
