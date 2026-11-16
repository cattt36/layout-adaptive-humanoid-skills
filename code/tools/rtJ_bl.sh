#!/bin/bash
# DEMO, NEW G1 (waist free), 5PR with a NARROWER zig-zag (user: change the stake spacing).  The vK 5PR fails
# by sideways falls: the human hops 0.65-0.85 m sideways between L and R stakes; 3PR (forward hops, works on the robot)
# vs 5PR under the site condition 30 % vs 9 % (bmJ/prcmp), no torque saturation in either.  tools/warp_5pr.py pulls
# R stakes 2 / 4 towards the L1-L5 line (feet order, heights, timing, run-up unchanged), stakes fixed +-2 cm, NO pushes,
# from rtJbx_018, 3000 its:
#   rtJbl_W80 : lateral x0.8   (data/std_w5_80)   gpu 1
#   rtJbl_W65 : lateral x0.65  (data/std_w5_65)   gpu 2
#   rtJbl_W50 : lateral x0.5   (data/std_w5_50)   gpu 3
# Eval: RP=rtJbl OD=bl SRO=<root> tools/rtJ_bn_eval.sh + tools/rtJ_norope.sh.  Log runs/night/rtJbl.log.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbl.log; }
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
for s in 80 65 50; do
  [ -f $D/std_w5_$s/FiveStepPoleRun/FiveStepPoleRun_018_motion.npz ] || \
    /root/gpufree-data/venvs/deploy/bin/python tools/warp_5pr.py $D/std_bvB $D/std_w5_$s 0.$s >> $N/rtJbl.log 2>&1
done
train() {  # tag gpu stakes-root
  local TAG=$1 G=$2 SR=$3
  local T=FiveStepPoleRun RUN=rtJbl_$TAG SRC=$(last FiveStepPoleRun rtJbx_018)
  if [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ]; then
    log "$RUN from $SRC (gpu $G, $IT its, $SR)"
    ( cd $H/runs && env $NEW $STK HNU_STAKES_ROOT=$SR HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G \
        $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T \
        --agent.run-name $RUN --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
        --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
    log "$RUN finished: $(last $T $RUN)"
  fi
  RP=rtJbl OD=bl SRO=$SR G=$G bash tools/rtJ_bn_eval.sh $TAG >> $N/rtJbl_eval.log 2>&1
  OD=bl bash tools/rtJ_norope.sh $TAG $H/deploy_exports/rtJbl/$RUN >> $N/rtJbl_eval.log 2>&1
  log "$RUN eval done"
}
log "rtJbl start (pushes off)"
train W80 1 $D/std_w5_80 & sleep 20
train W65 2 $D/std_w5_65 & sleep 20
train W50 3 $D/std_w5_50 &
wait
{ TAGS="W80 W65 W50" OD=bl bash tools/rtJ_bn_eval.sh summary; echo; TAGS="W80 W65 W50" OD=bl bash tools/rtJ_norope.sh summary
  for t in W80 W65 W50; do echo "$t: $(cat $B/bl/${t}_hop.txt | paste -sd' ')"; done; } > $B/bl/summary.md 2>&1
log "rtJbl done"
