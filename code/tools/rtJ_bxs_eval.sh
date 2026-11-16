#!/bin/bash
# DEMO: evaluate a self-calibrated stakes round (tools/rtJ_bxs.sh): export -> REP sim with the stakes at the
# training positions (traces) -> landings (prop_spread) -> stakes moved to the new mean landings (calib_stakes = the
# real-robot procedure) -> REP + DEMO on the re-calibrated stakes.  Out: robot_sim_check/bmJ/calib_bs<ROUND>/summary.md
ROUND=${ROUND:-1}; PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; S=$H/data/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
CAL=${CAL:-$B/calib}; O=$B/calib_bs$ROUND; mkdir -p $O
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
while ! grep -q "bs$ROUND train done" $N/rtF.log; do sleep 60; done
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_PERCEPTION=1 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --start-yaw 1 --no-base-rand"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
for cs in ThreeStepPoleRun_012:s93 ThreeStepPoleRun_014:s93 FiveStepPoleRun_018:s83; do
  c=${cs%%:*}; T=${c%_*}; n=${c##*_}; RUN=rtJbs${ROUND}_$n; E=$H/deploy_exports/rtJbs/$RUN; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $(last $T $RUN) $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  L0=$CAL/on_$c; TD=$O/trace_$c; rm -rf $TD; mkdir -p $TD
  HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $T --motion-dirs $L0 --latent-dir $L0 $REP \
    --out $O/train_${c}_REP.json > $O/train_${c}_REP.log 2>&1
  $PYD tools/prop_spread.py $E $L0/${c}_motion.npz $TD 12 $O/spread_$c > $O/spread_$c.md 2>&1
  L=$O/on_$c; mkdir -p $L; cp $L0/${c}_motion.npz $L0/${c}_z.npy $L/
  $PYD tools/calib_stakes.py $L0/${c}_motion.npz $L0/${c}_stakes.json $O/spread_$c.json $L/${c}_stakes.json > $O/on_${c}_calib.txt
  for k in REP DEMO; do [ $k = REP ] && A=$REP || A=$DEMO
    $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $T --motion-dirs $L --latent-dir $L $A \
      --out $O/on_${c}_$k.json > $O/on_${c}_$k.log 2>&1
  done
  log "bs$ROUND $c: $(tail -n 1 $E/parity.txt)"
done
$PYD - $O << 'PEOF' > $O/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
print("| clip | stakes at training positions REP | re-calibrated REP | re-calibrated DEMO | (all stakes stepped + no fall, %) |\n|---|---|---|---|---|")
for c in ("ThreeStepPoleRun_012", "ThreeStepPoleRun_014", "FiveStepPoleRun_018"):
    v = []
    for f in (f"train_{c}_REP", f"on_{c}_REP", f"on_{c}_DEMO"):
        p = f"{O}/{f}.json"
        v.append(f"{100 * np.mean([bool(x['success_v2']) for x in json.load(open(p))['rows']]):.0f}" if os.path.exists(p) else "-")
    print(f"| {c} | " + " | ".join(v) + " | |")
PEOF
log "bs$ROUND eval done"
