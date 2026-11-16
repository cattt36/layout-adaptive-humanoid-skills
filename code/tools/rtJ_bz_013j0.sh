#!/bin/bash
# DEMO ONLY (not for the paper).  User: the bridge task = weave around the 4 posts IN ORDER (reference sides
# L R L R); the whole-bridge "layouts" of tools/beam4.py changed the task -> withdrawn.  With the real bridge at the
# REFERENCE pose the flat trackers fall (013 REP 11 % success, bmJ/btopo).  Blind tracker + FIXED bridge is learnable
# (the earlier bridge-on run rtJbx had posts +-10 cm / bridge +-5 cm random -> not learnable blind):
#   real bridge = reference pose, post tops = reference + DZ (mocap: 013 +0.027, 005 +0.057), floor bars + braces,
#   NO ramp / leg (hidden), placement error only (bridge +-2 cm / posts +-2 cm, height +-1 cm), contact penalised.
#   From the flat trackers rtJbxf_<clip> (good tracking), 4000 its:
#   rtJbz_013j2 (+-2 cm)  rtJbz_013j0 (exact)  rtJbz_013w10 (+-2 cm, contact weight -10)
#   rtJbz_005j2 (+-2 cm)  rtJbz_005j0 (exact)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbz.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
IT=${IT:-4000}
# data: same motions, bridge json with the real heights and the ramp / leg hidden
R=$D/std_ctreal/CrouchedTraverse; mkdir -p $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
NEW="HNU_ACTOR_BLIND=1 HNU_START_FRAC=0.4 HNU_REST_FRAC=0.8 HNU_STEP_FRAC=0 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8
     HNU_REST_XY=0.02 HNU_REST_YAW=3 HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_DR=1 HNU_DR_X=1
     HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10
     HNU_ROPE=0.9,1.0,12.0,0.8 HNU_ROPE_LEN=2.0 HNU_REST_RP=3 HNU_EP_LEN=25 HNU_PUSH_SCALE=1.25
     HNU_GROOT_STD=0.4 HNU_ANCHOR_XY=1.0 HNU_ANCHOR_XY_BRIDGE=1.0"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_SLIP_W=-0.3 HNU_BRIDGE_ROOT=$D/std_ctreal"
J2="HNU_BRIDGE_JITTER_XY=0.02 HNU_BRIDGE_JITTER_ZLO=-0.01 HNU_BRIDGE_JITTER_ZHI=0.01 HNU_BRIDGE_POST_JITTER=0.02"
J0="HNU_BRIDGE_JITTER_XY=0 HNU_BRIDGE_JITTER_ZLO=0 HNU_BRIDGE_JITTER_ZHI=0 HNU_BRIDGE_POST_JITTER=0"
mkdir -p $S/lists
train() {  # clip run source-run gpu env...
  local C=$1 RUN=$2 SR=$3 G=$4; shift 4
  local T=CrouchedTraverse SRC=$(last CrouchedTraverse $SR) A=$S/lists/bz_CrouchedTraverse_$C.txt
  echo ${T}_$C > $A
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$RUN exists"; return; }
  log "$RUN from $SRC (gpu $G)"
  ( cd $H/runs && env "$@" HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_CLIPS_ALLOW=$A CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py \
      HNU-Track-$T-Bridge --env.scene.num-envs ${NENV:-4096} --agent.experiment-name hnu_track_$T --agent.run-name $RUN \
      --agent.max-iterations $IT --agent.resume True --agent.load-run $(basename $(dirname $SRC)) \
      --agent.load-checkpoint $(basename $SRC) > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$RUN finished: $(last $T $RUN)"
}
# 013j0 moved from GPU 1 (shared with another user) to GPU 0 (shared with rtJbz_013j2)
train 013 rtJbz_013j0 rtJbxf_013 0 $NEW $BASE $BRG $J0
log "bz 013j0 done"
