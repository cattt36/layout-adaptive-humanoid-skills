#!/bin/bash
# Stake curriculum for one task, starting from the flat expert (which drifts ~0.45 m from the reference's global
# path, so foot placement precision has to be learned before height):
#   stage   stake height  foot tolerance  anchor-xy limit
#   p1      0.03 m        0.30 m          0.60 m
#   p2      0.03 m        0.17 m          0.40 m
#   h06     0.06 m        0.17 m          0.40 m
#   h10     0.10 m        0.17 m          0.40 m   (ThreeStepPoleRun/FiveStepPoleRun.usd spec)
# Each stage resumes from the previous stage's last checkpoint and is evaluated (clean, push+noise) with the
# strict stake criterion at its height.
# usage: stakes_curriculum.sh <Task> <gpu> [iters_per_stage] [prefix] [start_ckpt] [stages...]
# (v2: prefix curr2, HNU_STAKE_RW>0 adds the direct foot-to-stake reward, start from the p1 checkpoint)
T=$1; G=$2; ITERS=${3:-3000}; PFX=${4:-curr}; START=${5:-}
STAGES=()
if [ $# -gt 5 ]; then shift 5; STAGES=("$@"); fi
[ ${#STAGES[@]} = 0 ] && STAGES=("p1 0.03 0.30 0.60" "p2 0.03 0.17 0.40" "h06 0.06 0.17 0.40" "h10 0.10 0.17 0.40")
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
L=$H/runs/logs/stakes_${PFX}_${T}.log
log() { echo "[$(date +%H:%M:%S)] $*" >> $L; }
SRC=${START:-$(cat $H/runs/expert_eval/$T/checkpoint.txt)}
for ST in "${STAGES[@]}"; do
  set -- $ST; TAG=$1; HS=$2; TOL=$3; AXY=$4
  # HSCALE < 1: robot-scale stride (stake spacing shortened to the robot's achievable stride), own data dirs
  SC=${HSCALE:-1.0}
  D=$H/data/ref_stakes_h${HS#0.}; [ "$SC" != "1.0" ] && D=$H/data/ref_stakes_s${SC#0.}_h${HS#0.}
  if [ ! -d $D/$T ]; then
    $PY stage1_reference/make_stakes.py --task $T --ref $H/data/ref_v1/$T --csv $H/data/scene/$T.csv --height $HS \
      --hscale $SC --out $D >> $L 2>&1
    rm -f $D/$T/ThreeStepPoleRun_007_*   # CSV anomaly, excluded as in ref_stakes
  fi
  export HNU_STAKE_H=$HS HNU_STAKE_TOL=$TOL HNU_ANCHOR_XY=$AXY HNU_STAKES_ROOT=$D
  RUN=${PFX}_$TAG
  if ! ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt > /dev/null 2>&1; then
    log "$T stage $TAG (h $HS, tol $TOL, axy $AXY) from $SRC"
    ( cd $H/runs && CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Stakes --env.scene.num-envs 4096 \
      --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $ITERS \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
  fi
  RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN | tail -1)
  SRC=$(ls $RD/model_*.pt | sort -V | tail -1)
  O=$H/runs/stakes_eval/${PFX}_$TAG/$T
  mkdir -p $O
  CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --stakes --ckpt $SRC --per-clip 8 --motion-dir $D/$T \
    --stake-top $HS --out $O > $O/clean.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --stakes --ckpt $SRC --per-clip 8 --motion-dir $D/$T \
    --stake-top $HS --push --noise --out $O > $O/push.log 2>&1
  log "$T stage $TAG done: $SRC  $(grep -a success_rate $O/clean.log | cut -c1-220)"
done
log "$T curriculum finished"
