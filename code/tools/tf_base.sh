#!/bin/bash
# teacher fine-tune for the real-robot deployment (user decision; paper models / results unchanged):
#  CrouchedTraverse: support-post bases of the real bridge added (HNU_POST_BASE=1: floor bar 100 x 10 x 3 cm + 2 braces
#     per post; they were there in the motion capture, missing in the sim).  From the deployed bridge teacher du3_B,
#     identical recipe (tools/du_ft_auto.sh, variant B) + bases.  Variants: A bar height 2.5-4 cm, B 2-5 cm.  4000 its.
#  FiveStepPoleRun: swing foot scuffing the floor between the stakes (diagnostic: teacher 4.6 touches / rollout).
#     From tf2_full, identical recipe (tools/tf_v2.sh W2) + floor_scuff penalty per foot and step while no reference
#     foot is at floor level (hnu_tasks/stakes.py).  Variants: C weight -2, D weight -5.  3000 its.
#  25 % frame-0 resets in all runs (retention).  Gate: tools/tf_base_gate.sh.
# usage: bash tools/tf_base.sh        log runs/night/tf_base.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/tf_base.log; }
CK=$(cat $N/v6g10n_teachers.txt)
get() { echo $CK | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$1" | tail -1; }
train() {  # task suffix gpu src run iters env...
  local T=$1 SUF=$2 G=$3 SRC=$4 RUN=$5 IT=$6; shift 6
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)"
}
BRG="HNU_DR=1 HNU_DR_LEVEL=2 HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_START_FRAC=0.25 HNU_PUSH_SCALE=1.0
     HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06
     HNU_BRIDGE_POST_JITTER=0.10 HNU_POST_BASE=1"
STK="HNU_DR=1 HNU_DR_LEVEL=1 HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8
     HNU_STAKE_RW=5 HNU_START_FRAC=0.25 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0
     HNU_STAKE_JITTER=0.08 HNU_STAKE_STRAIGHT=0.3 HNU_PUSH_SCALE=1.25 HNU_SLIP_W=-0.1
     HNU_STAKES_ROOT=$H/data/ref_stakes_s83_h10"
log "start: CT from $(get CrouchedTraverse), 5PR from $(get FiveStepPoleRun)"
train CrouchedTraverse -Bridge ${G_A:-3} $(get CrouchedTraverse) base_A 4000 $BRG HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01 &
sleep 30
train CrouchedTraverse -Bridge ${G_B:-4} $(get CrouchedTraverse) base_B 4000 $BRG HNU_BASE_DZ_LO=-0.01 HNU_BASE_DZ_HI=0.02 &
sleep 30
train FiveStepPoleRun -Stakes ${G_C:-3} $(get FiveStepPoleRun) scuff_C 3000 $STK HNU_SCUFF_W=-2.0 &
sleep 30
train FiveStepPoleRun -Stakes ${G_D:-4} $(get FiveStepPoleRun) scuff_D 3000 $STK HNU_SCUFF_W=-5.0 &
wait
log "all trained"
