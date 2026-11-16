#!/bin/bash
# DEMO: mid-training check of the blind demo trackers (rtJbx): latest checkpoint of a few runs -> export ->
# sim2sim REP (one robot: rope start, 1 cm / 1 deg placement, no base randomisation), 32 trials -> prop_spread.
# Usage: bash tools/bx_midcheck.sh <ckpt_iter_3PR> <out_tag>   (waits for model_<iter>.pt of rtJbxf_014)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
IT=$1; TAG=$2; O=$B/mid_$TAG; mkdir -p $O
while [ -z "$(ls $H/runs/logs/rsl_rl/hnu_track_ThreeStepPoleRun/*_rtJbxf_014/model_$IT.pt 2>/dev/null)" ]; do sleep 30; done
sleep 20
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_PERCEPTION=1 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' \
  HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
A="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --start-yaw 1 --no-base-rand"
for r in ThreeStepPoleRun:014:rtJbxf ThreeStepPoleRun:014:rtJbx CrouchedTraverse:005:rtJbx FiveStepPoleRun:018:rtJbxf; do
  T=$(echo $r | cut -d: -f1); C=$(echo $r | cut -d: -f2); P=$(echo $r | cut -d: -f3); RUN=${P}_$C
  CK=$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt | sort -V | tail -1); E=$H/deploy_exports/bxmid_$TAG/$RUN
  mkdir -p $E; CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $CK $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.log 2>&1
  if [ $P = rtJbxf ]; then L=$B/flat/${T}_$C; CR=track; SE="HNU_STAKES_ROOT=$D/std_flat"; BJ=""
  else L=$B/lay/${T}_$C; CR=task; SE=""; BJ=$(ls $L/*_bridge.json 2>/dev/null); fi
  [ -d $L ] || { mkdir -p $L; cp $D/std_flat/$T/${T}_${C}_motion.npz $L/; cp $S/z_tables_cv_f1/Z3/${T}_${C}_z.npy $L/; }
  TD=$O/trace_$RUN; rm -rf $TD; mkdir -p $TD
  env $SE HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit $CR --trials 32 --workers 32 --tasks $T --motion-dirs $L \
    --latent-dir $L $A --out $O/$RUN.json > $O/$RUN.log 2>&1
  $PYD tools/prop_spread.py $E $L/${T}_${C}_motion.npz $TD 12 $O/$RUN $BJ > $O/$RUN.md 2>&1
  echo "== $RUN $(basename $CK): $(tail -n 1 $O/$RUN.log)"
done
