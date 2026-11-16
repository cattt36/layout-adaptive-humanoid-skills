#!/bin/bash
# real-robot retraining, stage 1: teacher fine-tunes (paper models / results unchanged).
# Real G1, Deployment_E (model), D1: 0 / 13 -- falls right after the start.  The clips do not start at rest
# (D1 frame 0: pelvis 0.32 m/s, joints up to 2 rad/s, turning), training / sim2sim started every rollout WITH these
# velocities; the robot starts at rest (rope) or from the walking controller stepping in place.  Sim2sim of the vE
# model, 48 trials: start at rest + 1 s hold D1 69 / T1 40 %, stepping + hold + real-world deviations D1 40 / D2 17 / S1 40
# / T1 10 % (robot_sim_check/out_start_E).
# New in every teacher (from the deployed teachers du3_B / rs10c / tf2_full):
#   start from rest: 40 % of the resets at frame 0, 60 % of those at rest (frame-0 pose or stepping-in-place state,
#     data/step_states/step_states_v1.npz), start offset +-10 cm / +-15 deg, reference clock held 0.2-2 s (p 0.8);
#   contact / actuator randomisation HNU_DR_X (torsional friction 0-0.05 m, friction 0.3-1.5, soft soles, payload
#     0-1.5 kg, ankle effort 35-50 Nm), DR level 2 (PD +-20 % per joint);
#   venue of bars cut (inner posts 15 cm, end posts 20 cm per side from the post axis), stakes 40 cm / 10 cm.
# 25 % ... frame-0 resets with the reference velocities are kept (retention).  3000 its.  Log runs/night/rtF.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 PYTHONPATH=$H/code
NEW="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.6 HNU_STEP_FRAC=0.5 HNU_HOLD=0.2,2.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.10 HNU_REST_YAW=15
     HNU_STEP_STATES=$D/step_states/step_states_v1.npz HNU_DR=1 HNU_DR_X=1 HNU_DR_LEVEL=2 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1
     HNU_STAKE_R=0.20 HNU_STAKE_H=0.10"
BASE="HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
BRG="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_PUSH_SCALE=1.0 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05
     HNU_BRIDGE_JITTER_ZLO=-0.08 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10"
STK="HNU_PERCEPTION=1 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 HNU_STAKE_RW=5 HNU_STAKE_SKIP_FIRST=0
     HNU_STAKE_CONTACT=1 HNU_STK_FULL=2.0 HNU_STK_EDGE=0.5 HNU_STK_MISS=-2.0 HNU_STAKE_STRAIGHT=0.3 HNU_SLIP_W=-0.1"
W1A=$D/clip_split_trial/teacher_w1_allow.txt
W1="HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_PUSH_SCALE=1.0 HNU_CLIPS_ALLOW=$W1A
    HNU_STAKES_ROOT=$D/ref_stakes_s93_h10"
W2="HNU_STAKE_JITTER=0.08 HNU_PUSH_SCALE=1.25 HNU_STAKES_ROOT=$D/ref_stakes_s83_h10"
CUR=$(cat $N/v6g10n_teachers.txt)
get() { echo $CUR | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$1" | tail -1; }
train() {  # task suffix gpu src run iters env...
  local T=$1 SUF=$2 G=$3 SRC=$4 RUN=$5 IT=$6; shift 6
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && { log "$T $RUN exists"; return; }
  ( cd $H/runs && env "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  log "$T $RUN finished: $(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)"
}
log "teachers start: CT $(get CrouchedTraverse) 3PR $(get ThreeStepPoleRun) 5PR $(get FiveStepPoleRun)"
IT=${IT:-3000}
train CrouchedTraverse -Bridge 0 $(get CrouchedTraverse) rtF $IT $NEW $BASE $BRG HNU_BASE_MID="$MID" HNU_BASE_END="$END" &
sleep 20
train ThreeStepPoleRun -Stakes 1 $(get ThreeStepPoleRun) rtF $IT $NEW $STK $W1 &
sleep 20
train FiveStepPoleRun -Stakes 2 $(get FiveStepPoleRun) rtF $IT $NEW $STK $W2 &
wait
log "teachers done"
