#!/bin/bash
# DEMO, NEW G1 (waist free), 5PR round 3, ALL WITHOUT PUSHES (no pushes on site).  Round bh (bmJ/bh):
# the feet-height reward raises the hop to the reference (left apex 0.26 -> 0.39-0.44 m) but stake-1 accuracy collapses
# (REP 0-16 %); no-push without it (Bp0) is the most robust so far (BASE 55 / L40 44 / DRX 33 vs old 30 / 31 / 12).
#   rtJbh_Bp0c   : Bp0 continued 3000 its                                                   gpu 0
#   rtJbh_Bj6p0  : option B +-6 cm, no push, no height reward, from rtJbx_018            gpu 1
#   rtJbh_B1p0   : fixed +-2 cm, no push, feet height weight 1 / std 0.08 (gentle)         gpu 2
#   rtJbh_B3p0c  : B3p0 continued 3000 its (does the high hop get accurate again)           gpu 3
#   rtJbh_Bj6p0c : rtJbn_Bj6 (push 1.0, best of round bn) continued 3000 its, no push      gpu 4
# Eval: RP=rtJbh OD=bh tools/rtJ_bn_eval.sh + tools/rtJ_norope.sh.  Log runs/night/rtJbh2.log.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbh2.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-3000}
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=0
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
  RP=rtJbh OD=bh G=$G bash tools/rtJ_bn_eval.sh $TAG >> $N/rtJbh2_eval.log 2>&1
  OD=bh bash tools/rtJ_norope.sh $TAG $H/deploy_exports/rtJbh/$RUN >> $N/rtJbh2_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbh2 start (pushes off)"
train Bp0c rtJbh_Bp0 0 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB & sleep 20
train Bj6p0 rtJbx_018 1 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_STAKE_JITTER=0.06 & sleep 20
train B1p0 rtJbx_018 2 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_FEET_Z_W=1 HNU_FEET_Z_STD=0.08 & sleep 20
train B3p0c rtJbh_B3p0 3 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_FEET_Z_W=3 & sleep 20
train Bj6p0c rtJbn_Bj6 4 $NEW $STK HNU_STAKES_ROOT=$D/std_bvB HNU_STAKE_JITTER=0.06 &
wait
TG="old Bp0 B3p0 Bp0c Bj6p0 B1p0 B3p0c Bj6p0c"
{ TAGS="$TG" OD=bh bash tools/rtJ_bn_eval.sh summary; echo; TAGS="$TG" OD=bh bash tools/rtJ_norope.sh summary; } > $B/bh/summary2.md 2>&1
log "rtJbh2 done"
