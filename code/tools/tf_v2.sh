#!/bin/bash
# Teacher fine-tune for success criterion v2.  From the current teachers (v6g10t bank).
#  ThreeStepPoleRun / FiveStepPoleRun (same base recipe as tools/robust_stakes_h10c.sh) + contact-based stake objective (one payout per stake:
#     fully on +2 / edge +0.5 / missed -2 and termination), first stake NOT relaxed, stake jitter +-8 cm, straight
#     W1 layouts in 30 % of the resets, pushes x1.25, mild foot-slip penalty.
#  CrouchedTraverse (same base recipe as run_v6r2.sh r2_full_bridge) + graded bridge-contact penalty (light touch cheap,
#     hard hit expensive), foot-slip penalty, pushes x1.3.
#  25 % of the resets at frame 0 (retention of the start of the motion).
#  usage: tf_v2.sh <run-tag> <iters> [W1|W2|DU ...]
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
TAG=$1; IT=$2; shift 2
# source checkpoints: SRC_FILE (default the current v6g10t bank; e.g. runs/tf2_pilot_teachers.txt to continue a pilot)
src() { sed 's/ /\n/g' ${SRC_FILE:-$H/runs/night/v6g10t_teachers.txt} | grep -A1 -- "--ckpt-$1" | tail -1; }
train() {  # task suffix gpu src run iters env...
  local T=$1 SUF=$2 G=$3 SRC=$4 RUN=$5 N=$6; shift 6
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { echo "$T $RUN exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $N \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  echo "$T $RUN finished: $(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)"
}
STK="HNU_DR=1 HNU_DR_LEVEL=1 HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8
     HNU_STAKE_RW=5 HNU_START_FRAC=0.25 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0
     HNU_STAKE_JITTER=0.08 HNU_STAKE_STRAIGHT=0.3 HNU_PUSH_SCALE=1.25 HNU_SLIP_W=-0.1"
BRG="HNU_DR=1 HNU_DR_LEVEL=2 HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_START_FRAC=0.25
     HNU_PUSH_SCALE=1.3 HNU_SLIP_W=-0.3"
# W1 round 2: long W1 strides (1.5-1.8 m) are close to the G1 jumping limit -> no along-course
# stretching (jitter along +-3 cm, across +-6 cm), pushes back to the default (real deployment has no external
# pushes), the two data-quality exclusions (ThreeStepPoleRun_022 / ThreeStepPoleRun_025) removed from the teacher data
W1B_ALLOW=$H/data/clip_split_trial/teacher_w1_allow.txt
ls $H/data/ref_stakes_s93_h10/ThreeStepPoleRun/*_motion.npz | xargs -n1 basename | sed 's/_motion.npz//' \
  | grep -v -e ThreeStepPoleRun_022 -e ThreeStepPoleRun_025 > $W1B_ALLOW
W1B="HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_PUSH_SCALE=1.0 HNU_CLIPS_ALLOW=$W1B_ALLOW"
for X in "$@"; do
  case $X in
    W1B) train ThreeStepPoleRun -Stakes 0 $(src ThreeStepPoleRun) tf2_$TAG $IT $STK $W1B HNU_STAKES_ROOT=$H/data/ref_stakes_s93_h10 & ;;
    W1) train ThreeStepPoleRun -Stakes 0 $(src ThreeStepPoleRun) tf2_$TAG $IT $STK HNU_STAKES_ROOT=$H/data/ref_stakes_s93_h10 & ;;
    W2) train FiveStepPoleRun -Stakes 1 $(src FiveStepPoleRun) tf2_$TAG $IT $STK HNU_STAKES_ROOT=$H/data/ref_stakes_s83_h10 & ;;
    DU) train CrouchedTraverse -Bridge ${DU_GPU:-0} $(src CrouchedTraverse) tf2_$TAG $IT $BRG & ;;
  esac
done
wait
