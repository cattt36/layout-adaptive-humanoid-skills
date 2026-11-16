#!/bin/bash
# DEMO: evaluate rtJby (tools/rtJ_by.sh) like the real-robot procedure: export -> REP sim with the stakes at the
# current calibrated positions (bmJ/calib/on_<clip>, traces) -> landings (prop_spread) -> stakes moved to the new mean
# landings (calib_stakes) -> REP, DEMO and REP 6 deg heading error compensated (+-1 deg) on the re-calibrated stakes.
# Out: robot_sim_check/bmJ/calib_by/summary.md, "by eval done" in runs/night/rtJby.log
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
CAL=$B/calib; O=$B/calib_by; mkdir -p $O
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJby.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
while ! grep -q "by train done" $N/rtJby.log; do sleep 60; done
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_PERCEPTION=1 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
for rc in by_016a:FiveStepPoleRun_016 by_016f:FiveStepPoleRun_016 by_016b:FiveStepPoleRun_016 by_018a:FiveStepPoleRun_018 \
          by_014a:ThreeStepPoleRun_014; do
  r=${rc%%:*}; c=${rc#*:}; T=${c%_*}; RUN=rtJ$r; E=$H/deploy_exports/rtJby/$RUN; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $(last $T $RUN) $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  L0=$CAL/on_$c; TD=$O/trace_$r; rm -rf $TD; mkdir -p $TD
  HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $T --motion-dirs $L0 --latent-dir $L0 $REP \
    --start-yaw 1 --out $O/train_${r}_REP.json > $O/train_${r}_REP.log 2>&1
  $PYD tools/prop_spread.py $E $L0/${c}_motion.npz $TD 12 $O/spread_$r > $O/spread_$r.md 2>&1
  L=$O/on_$r; mkdir -p $L; cp $L0/${c}_motion.npz $L0/${c}_z.npy $L/
  $PYD tools/calib_stakes.py $L0/${c}_motion.npz $L0/${c}_stakes.json $O/spread_$r.json $L/${c}_stakes.json > $O/on_${r}_calib.txt
  for k in REP DEMO y6c; do
    case $k in REP) A="$REP --start-yaw 1";; DEMO) A=$DEMO;; y6c) A="$REP --start-yaw 6 --yaw-comp --yaw-meas-noise 1";; esac
    $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $T --motion-dirs $L --latent-dir $L $A \
      --out $O/on_${r}_$k.json > $O/on_${r}_$k.log 2>&1
  done
  log "by $r: $(tail -n 1 $E/parity.txt)"
done
$PYD - $O << 'PEOF' > $O/summary.md
import json, os, sys
import numpy as np
O = sys.argv[1]
print("| run | stakes at old calib REP | re-calibrated REP | REP no fall | re-calibrated DEMO | REP 6 deg comp +-1 | (all stakes stepped + no fall, %) |\n|---|---|---|---|---|---|---|")
for r in ("by_016a", "by_016f", "by_016b", "by_018a", "by_014a"):
    v = []
    for f, key in ((f"train_{r}_REP", 0), (f"on_{r}_REP", 0), (f"on_{r}_REP", 1), (f"on_{r}_DEMO", 0), (f"on_{r}_y6c", 0)):
        p = f"{O}/{f}.json"
        if not os.path.exists(p):
            v.append("-"); continue
        rows = json.load(open(p))["rows"]
        v.append(f"{100 * np.mean([bool(x['success_v2']) if key == 0 else x.get('cause') != 'fall' for x in rows]):.0f}")
    print(f"| {r} | " + " | ".join(v) + " | |")
PEOF
log "by eval done"
