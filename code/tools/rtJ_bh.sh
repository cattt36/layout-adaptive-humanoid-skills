#!/bin/bash
# DEMO, NEW G1 (waist free), 5PR round 2.  rtJbn (fixed stakes / push 1.0) did NOT beat the vK model:
# REP 11-56 % vs 74 %, all variants fragile (bmJ/bn).  The vK model hops ~13 cm lower than the reference with the left
# foot (tools/hop_height.py: apex 0.25 vs 0.38 m) -- the world-frame feet reward is saturated by the blind drift, so nothing
# asks for the hop height.  New opt-in reward HNU_FEET_Z_W (foot HEIGHT only, std HNU_FEET_Z_STD 0.05 m):
#   rtJbh_B3   : option B fixed +-2 cm, push 1.0, feet height weight 3      from rtJbx_018       gpu 0
#   rtJbh_Bj3  : option B +-6 cm (as vK),  push 1.0, feet height 3       from rtJbx_018       gpu 1
#   rtJbh_B6   : option B fixed +-2 cm, push 1.0, feet height 6            from rtJbx_018       gpu 2
#   rtJbh_A3   : human pattern fixed +-2 cm, push 1.0, feet height 3       from rtJbx_018       gpu 3
#   rtJbh_Bj6c : rtJbn_Bj6 continued 3000 its (no height reward)                                gpu 4
# Eval: RP=rtJbh OD=bh tools/rtJ_bn_eval.sh (+ hop height).  Log runs/night/rtJbh.log.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbh.log; }
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
A=$S/lists/bw_FiveStepPoleRun_018.txt
train() {  # tag source-run gpu env...
  local TAG=$1 SR=$2 G=$3; shift 3
  local T=FiveStepPoleRun RUN=rtJbh_$TAG SRC=$(last FiveStepPoleRun $SR)
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  RP=rtJbh OD=bh G=$G bash tools/rtJ_bn_eval.sh $TAG >> $N/rtJbh_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbh start"
train B3 rtJbx_018 0 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_FEET_Z_W=3 & sleep 20
train Bj3 rtJbx_018 1 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_FEET_Z_W=3 HNU_STAKE_JITTER=0.06 & sleep 20
train B6 rtJbx_018 2 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_FEET_Z_W=6 & sleep 20
train A3 rtJbx_018 3 $NEW $STK HNU_STAKES_ROOT=$S/ref_stakes_s83_h10 HNU_FEET_Z_W=3 & sleep 20
train Bj6c rtJbn_Bj6 4 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_STAKE_JITTER=0.06 &
wait
OD=bh TAGS="B3 Bj3 B6 A3 Bj6c" bash tools/rtJ_bn_eval.sh summary > $B/bh/summary.md 2>&1
log "rtJbh done"
