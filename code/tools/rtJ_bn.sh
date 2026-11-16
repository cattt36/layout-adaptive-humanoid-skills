#!/bin/bash
# DEMO, NEW G1 (waist free): the vK 5PR tracker rtJbx_018 failed on the robot (first stake missed, fall at
# ~5.5 s).  sim (bmJ/abl5): nominal REP 83 % but every single realistic perturbation halves it -- friction / CoM /
# encoder bias 48 %, fixed 20 ms lag 52 %, 0-40 ms 44 %, actuator / contact DR 16 %, 3 deg start yaw uncompensated 22 %.
# Retrain with the recipe that worked for the faulty-waist robot (rtJbv2_018Ap): stakes FIXED where they stand (+-2 cm),
# pushes x1.0, from rtJbx_018, 3000 its:
#   rtJbn_B    : vK option-B layout (stakes already on site), push 1.0                     gpu 0
#   rtJbn_Bp   : same, push 1.25 (control)                                                    gpu 1
#   rtJbn_A    : human pattern, push 1.0                                                      gpu 2
#   rtJbn_Bj6  : option B, stakes +-6 cm as in vK, push 1.0 (isolates the push effect)      gpu 3
#   rtJbn_By5  : option B, push 1.0, start placement +-3 cm / +-5 deg (vK: 2 cm / 3 deg)      gpu 4
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbn.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-3000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.0
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1
     HNU_STAKE_JITTER=0.02"
A=$S/lists/bw_FiveStepPoleRun_018.txt; echo FiveStepPoleRun_018 > $A
train() {  # tag gpu env...
  local TAG=$1 G=$2; shift 2
  local T=FiveStepPoleRun RUN=rtJbn_$TAG SRC=$(last FiveStepPoleRun rtJbx_018)
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  G=$G bash tools/rtJ_bn_eval.sh $TAG >> $N/rtJbn_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbn start"
train B 0 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB & sleep 20
train Bp 1 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_PUSH_SCALE=1.25 & sleep 20
train A 2 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 & sleep 20
train Bj6 3 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_STAKE_JITTER=0.06 & sleep 20
train By5 4 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_REST_XY=0.03 HNU_REST_YAW=5 &
wait
bash tools/rtJ_bn_eval.sh summary > $B/bn/summary.md 2>&1
log "rtJbn done"
