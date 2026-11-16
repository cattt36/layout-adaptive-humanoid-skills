#!/bin/bash
# DEMO ONLY: after tools/rtJ_bm_post.sh -- where the trackers actually go (prop placement): sim2sim traces
# of every tracker on its own clip in the DEMO condition (rope 80 % BW over 10 s, 12 s hold, start 2 cm / 3 deg, contact /
# actuator / payload randomisation, 0-40 ms, no pushes) and REP (one robot, run-to-run repeatability: same rope start and
# placement 1 cm / 1 deg (floor foot template; sim2sim start-yaw = fixed magnitude, random sign), no contact / mass / encoder randomisation, 0-20 ms), 64 trials each, then tools/prop_spread.py (pelvis deviation, foot
# landings, post clearance in the start frame).  Out: robot_sim_check/bmJ/spread/<run>.md/.json + spread.md,
# log "bx spread done"
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
while [ $(grep -c "bx post done" $N/rtF.log) -lt ${NPOST:-2} ]; do sleep 60; done
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_PERCEPTION=1 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' \
  HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_ROPE_LEN=2.0
O=$B/spread_bx; mkdir -p $O
declare -A A=([DEMO]="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
           [REP]="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --start-yaw 1 --no-base-rand")
for E in $H/deploy_exports/rtJbx/rtJbx*_*/; do E=${E%/}
  RUN=$(basename $E); C=${RUN##*_}; T=$($PYD -c "import json; print(json.load(open('$E/policy_meta.json'))['tracker']['task'])")
  # flat CT: clearance to the REFERENCE posts (where the props would stand); stakes: no bridge json -> none
  if [[ $RUN == *bxf_* ]]; then L=$B/flat/${T}_$C; CR=track; BJ=$(ls $B/lay/${T}_$C/*_bridge.json 2>/dev/null); else L=$B/lay/${T}_$C; CR=task; BJ=$(ls $L/*_bridge.json 2>/dev/null); fi
  for c in REP DEMO; do
    TD=$O/trace_${RUN}_$c; mkdir -p $TD
    [ -f $O/${RUN}_$c.json ] || HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit $CR --trials 64 --workers 100 --tasks $T \
      --motion-dirs $L --latent-dir $L ${A[$c]} --out $O/${RUN}_$c.json > $O/${RUN}_$c.log 2>&1
    $PYD tools/prop_spread.py $E $L/${T}_${C}_motion.npz $TD 12 $O/${RUN}_$c $BJ > $O/${RUN}_$c.md 2>&1
  done
done
cat $O/rtJbx*.md > $O/spread.md
log "bx spread done"
