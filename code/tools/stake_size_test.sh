#!/bin/bash
# (user request, one-off check): real stakes of diameter 40 cm / height 15 cm instead of 30 / 10 cm (training).
# Deployed student (v6g10xm fold 1 seed 5 L3) exported with the stake size of each case; CPU sim2sim on the 6 planned
# stake layouts (v6), conditions A (nominal) / B (+DR), 100 trials, criterion v2, floor-touch log.  Stake height perceived
# correctly (mocap: stake-top centre).  Cases: r15h10 (training size), r20h10, r15h15, r20h15.
# Out: robot_sim_check/out_stakesize/<case>/<layout>_<cond>.json, summary robot_sim_check/out_stakesize/summary.md
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_stakesize
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 \
  HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 \
  HNU_POST_SHAPE=mesh12
mkdir -p $O
for C in r15h10 r20h10 r15h15 r20h15; do
  R=0.${C:1:2}; HH=0.${C:4:2}
  E=$H/deploy_exports/stakesize_$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_STAKE_R=$R HNU_STAKE_H=$HH CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py \
    --student $H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  mkdir -p $O/$C
  for L in $RS/v6/in/planned_layouts/[STN][12]_*Step*; do
    n=$(basename $L); t=$(echo $n | cut -d_ -f2)
    for c in A B; do
      X=""; [ $c = B ] && X="--dr"
      [ -f $O/$C/${n}_$c.json ] && continue
      $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 100 --tasks $t --motion-dirs $L \
        --latent-dir $L $X --floor-log --stake-h-meta --out $O/$C/${n}_$c.json > $O/$C/${n}_$c.log 2>&1
    done
  done
done
$PYD - << 'EOF' > $O/summary.md
import json, os, numpy as np
O = "/root/gpufree-data/robot_sim_check/out_stakesize"
cases = ["r15h10", "r20h10", "r15h15", "r20h15"]
lays = sorted({f.rsplit("_", 1)[0] for f in os.listdir(f"{O}/r15h10") if f.endswith(".json")})
print("# Stake size test (deployed L3, sim2sim, success v2 % A / B, floor touches per rollout A)\n")
print("| layout | " + " | ".join(f"{c} (r = {int(c[1:3])} cm, h = {int(c[4:6])} cm)" for c in cases) + " |")
print("|---|" + "---|" * len(cases))
for l in lays:
    cells = []
    for c in cases:
        v = []
        for cond in ("A", "B"):
            f = f"{O}/{c}/{l}_{cond}.json"
            v.append(json.load(open(f))["rows"] if os.path.exists(f) else None)
        if v[0] is None:
            cells.append("-"); continue
        s = [100 * np.mean([r["success_v2"] for r in x]) if x else float("nan") for x in v]
        ev = np.mean([len(r.get("floor_events", [])) for r in v[0]]) / 4.0  # ~4 contact rows per touch event (rough)
        fails = {}
        for r in v[0]:
            k = r["cause"] or ("missed stake" if not r["success_v2"] else None)
            if k: fails[k] = fails.get(k, 0) + 1
        cells.append(f"{s[0]:.0f} / {s[1]:.0f} (fail A: {', '.join(f'{k} {n}' for k, n in sorted(fails.items())) or '-'})")
    print(f"| {l} | " + " | ".join(cells) + " |")
EOF
echo done >> $O/summary.md
