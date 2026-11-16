#!/bin/bash
# DEMO: evaluate rtJbz (bridge-on trackers, real bridge at the reference pose): export, REP / DEMO / REP 6 deg
# compensated with the training bridge json, weaving check (tools/ct_topo.py), contacts -> bmJ/bz/summary.md
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/bz; mkdir -p $O
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtJbz.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
while ! { grep -q "bz train done" $N/rtJbz.log && grep -q "bz 013j0 done" $N/rtJbz.log; }; do sleep 60; done
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_PERCEPTION=1 HNU_POST_BASE=1 \
  HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' \
  HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
R=$D/std_ctreal/CrouchedTraverse
for r in 013j2 013j0 013w10 005j2 005j0; do
  c=${r:0:3}; RUN=rtJbz_$r; E=$H/deploy_exports/rtJbz/$RUN; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $(last CrouchedTraverse $RUN) CrouchedTraverse $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  L=$O/dir_$c; mkdir -p $L
  cp $R/CrouchedTraverse_${c}_motion.npz $R/CrouchedTraverse_${c}_bridge.json $L/; cp $D/std/z_tables_cv_f1/Z3/CrouchedTraverse_${c}_z.npy $L/
  for k in REP DEMO y6c; do
    case $k in REP) A="$REP --start-yaw 1";; DEMO) A=$DEMO;; y6c) A="$REP --start-yaw 6 --yaw-comp --yaw-meas-noise 1";; esac
    TD=$O/trace_${r}_$k; rm -rf $TD; mkdir -p $TD
    HNU_S2S_TRACE=$TD $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 64 --tasks CrouchedTraverse --motion-dirs $L --latent-dir $L $A \
      --out $O/${r}_$k.json > $O/${r}_$k.log 2>&1
    $PYD tools/ct_topo.py $E $L/CrouchedTraverse_${c}_motion.npz $TD 12 $L/CrouchedTraverse_${c}_bridge.json > $O/${r}_${k}_topo.txt 2>&1
  done
  log "bz eval $r: $(tail -n 1 $E/parity.txt)"
done
$PYD - $O << 'PEOF' > $O/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
print("| run | cond | no fall % | completed % | runs with contact % | contacts per run | weaving (robot runs) |\n|---|---|---|---|---|---|---|")
for r in ("013j2", "013j0", "013w10", "005j2", "005j0"):
    for k in ("REP", "DEMO", "y6c"):
        f = f"{O}/{r}_{k}.json"
        if not os.path.exists(f):
            continue
        rows = json.load(open(f))["rows"]
        nf = np.mean([x["cause"] != "fall" for x in rows]); ok = np.mean([x["cause"] is None for x in rows])
        anyc = np.mean([sum(x.get("bridge_contacts", {}).values()) > 0 for x in rows])
        agg = {}
        for x in rows:
            for kk, v in x.get("bridge_contacts", {}).items():
                kk = kk.replace("bridge_", "").rstrip("0123456789")
                agg[kk] = agg.get(kk, 0) + v / len(rows)
        t = f"{O}/{r}_{k}_topo.txt"
        w = [l.split(":", 1)[1].strip() for l in open(t) if l.startswith("robot runs")] if os.path.exists(t) else []
        print(f"| {r} | {k} | {100*nf:.0f} | {100*ok:.0f} | {100*anyc:.0f} | " + ", ".join(f"{a} {b:.1f}" for a, b in sorted(agg.items())) + f" | {w[0] if w else '-'} |")
PEOF
log "bz eval done"
