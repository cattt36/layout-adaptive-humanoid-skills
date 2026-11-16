#!/bin/bash
# DEMO, NEW G1 (waist free), 5PR round 4 (user: on the robot the FIRST hop does not take off, the policy is the
# problem).  tools/first_hop.py (bmJ/fhop): in the hop onto stake 1 the vK model lifts its pelvis 5 cm vs 13 cm in the
# reference (it only lifts the leg: foot apex 0.26 vs 0.27 m), the push-off joints are far from their limits (|tau| <= 0.65-0.84
# of the limit, 0 % at the limit) -> nothing asks for the body to rise: the 3D anchor reward is saturated by the blind xy
# drift.  New opt-in reward HNU_ANCHOR_Z_W (anchor HEIGHT only, std HNU_ANCHOR_Z_STD).  Option B stakes fixed +-2 cm, NO
# pushes, 3000 its:
#   rtJbk_Z2   : anchor height w 2, std 0.05            from rtJbx_018      gpu 0
#   rtJbk_Z4   : anchor height w 4, std 0.05            from rtJbx_018      gpu 1
#   rtJbk_Z2w  : anchor height w 2, std 0.10 (wide)     from rtJbx_018      gpu 2
#   rtJbk_Z2b  : anchor height w 2, std 0.05            from rtJbh_Bp0      gpu 3
#   rtJbk_Z2j  : anchor height w 2, std 0.05, stakes +-6 cm                    gpu 4
# Starts when round 3 (rtJbh2) has finished training.  Eval: rtJ_bn_eval.sh + rtJ_norope.sh + first_hop.sh.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbk.log; }
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
  local T=FiveStepPoleRun RUN=rtJbk_$TAG SRC=$(last FiveStepPoleRun $SR)
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its)"
    ( cd $H/runs && env $NEW $STK HNU_STAKES_ROOT=$D/std_bvB "$@" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  RP=rtJbk OD=bk G=$G bash tools/rtJ_bn_eval.sh $TAG >> $N/rtJbk_eval.log 2>&1
  OD=bk bash tools/rtJ_norope.sh $TAG $H/deploy_exports/rtJbk/$RUN >> $N/rtJbk_eval.log 2>&1
  FHO=bk/fhop_$TAG ONLY5=1 E5=$H/deploy_exports/rtJbk/$RUN bash tools/first_hop.sh >> $N/rtJbk_eval.log 2>&1
  log "$RUN eval done"
}
until [ "$(grep -c ' finished: ' $N/rtJbh2.log 2>/dev/null)" -ge 5 ]; do sleep 30; done
log "rtJbk start (pushes off, anchor height reward)"
train Z2 rtJbx_018 0 HNU_ANCHOR_Z_W=2 & sleep 20
train Z4 rtJbx_018 1 HNU_ANCHOR_Z_W=4 & sleep 20
train Z2w rtJbx_018 2 HNU_ANCHOR_Z_W=2 HNU_ANCHOR_Z_STD=0.10 & sleep 20
train Z2b rtJbh_Bp0 3 HNU_ANCHOR_Z_W=2 & sleep 20
train Z2j rtJbx_018 4 HNU_ANCHOR_Z_W=2 HNU_STAKE_JITTER=0.06 &
wait
TG="Z2 Z4 Z2w Z2b Z2j"
{ TAGS="old $TG" OD=bk bash tools/rtJ_bn_eval.sh summary; echo; TAGS="$TG" OD=bk bash tools/rtJ_norope.sh summary
  for t in $TG; do echo "== $t"; grep -A1 "tr_5PR" $B/bk/fhop_$t/summary.txt | grep -v -- "--"; done; } > $B/bk/summary.md 2>&1
log "rtJbk done"
