#!/bin/bash
# DEMO: warped flat CT trackers (rtJbxw_*) on the ORIGINAL four-post bridge (reference positions, original
# heights, post floor bars + braces): REP / DEMO, falls and contacts.  Out: robot_sim_check/bmJ/bxw/summary.md
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/bxw; mkdir -p $O
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
while ! grep -q "bxw train done" $N/rtF.log; do sleep 60; done
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_PERCEPTION=1 HNU_POST_BASE=1 \
  HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' \
  HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --start-yaw 1 --no-base-rand"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
for c in 005 013; do
  RUN=rtJbxw_$c; E=$H/deploy_exports/rtJbx/$RUN; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $(last CrouchedTraverse $RUN) CrouchedTraverse $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  L=$O/bridge_$c; mkdir -p $L; cp $D/std_flatct2/CrouchedTraverse/CrouchedTraverse_${c}_motion.npz $L/
  cp $B/lay/CrouchedTraverse_$c/CrouchedTraverse_${c}_bridge.json $L/; cp $S/z_tables_cv_f1/Z3/CrouchedTraverse_${c}_z.npy $L/
  F=$O/flat_$c; mkdir -p $F; cp $L/CrouchedTraverse_${c}_motion.npz $L/CrouchedTraverse_${c}_z.npy $F/
  for k in REP DEMO; do [ $k = REP ] && A=$REP || A=$DEMO
    TD=$O/trace_${c}_$k; rm -rf $TD; mkdir -p $TD
    HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks CrouchedTraverse --motion-dirs $L --latent-dir $L $A \
      --out $O/bridge_${c}_$k.json > $O/bridge_${c}_$k.log 2>&1
    $PYD deploy/sim2sim.py $E --crit track --trials 64 --workers 100 --tasks CrouchedTraverse --motion-dirs $F --latent-dir $F $A \
      --out $O/flat_${c}_$k.json > $O/flat_${c}_$k.log 2>&1
  done
  $PYD tools/prop_spread.py $E $L/CrouchedTraverse_${c}_motion.npz $O/trace_${c}_REP 12 $O/spread_${c}_REP $L/CrouchedTraverse_${c}_bridge.json > $O/spread_${c}_REP.md 2>&1
  $PYD tools/prop_figure.py $O/spread_${c}_REP $L/CrouchedTraverse_${c}_motion.npz $O/bridge_${c}_REP.png $L/CrouchedTraverse_${c}_bridge.json > /dev/null 2>&1
  log "bxw $c: $(tail -n 1 $E/parity.txt)"
done
$PYD - $O << 'PEOF' > $O/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
print("| run | no fall % | completed % | runs with contact % | contacts per run |\n|---|---|---|---|---|")
for f in sorted(glob.glob(f"{O}/*_REP.json") + glob.glob(f"{O}/*_DEMO.json")):
    if "spread" in f:
        continue
    rows = json.load(open(f))["rows"]
    nf = np.mean([r["cause"] != "fall" for r in rows]); ok = np.mean([r["cause"] is None for r in rows])
    anyc = np.mean([sum(r.get("bridge_contacts", {}).values()) > 0 for r in rows])
    agg = {}
    for r in rows:
        for k, v in r.get("bridge_contacts", {}).items():
            g = "post" if k.startswith("post") else ("beam" if k == "plank" else "bar/brace")
            agg[g] = agg.get(g, 0) + v / len(rows)
    print(f"| {os.path.basename(f)[:-5]} | {100*nf:.0f} | {100*ok:.0f} | {100*anyc:.0f} | " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(agg.items())) + " |")
PEOF
log "bxw eval done"
