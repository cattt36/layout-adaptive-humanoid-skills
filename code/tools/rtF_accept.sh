#!/bin/bash
# sim2sim acceptance of the real-robot retraining (CPU MuJoCo, planned layouts, cut bars + 40 cm stakes,
# 64 trials, success v2).  Conditions:
#   A   start as before (frame 0 with the reference velocities), 0-20 ms        (comparison with the numbers)
#   R   at rest in the frame-0 pose (robot set down, no rope) + 1 s hold;  R3 the same with a 3 s hold
#   W   stepping-in-place state + 1 s hold                                       (walking controller -> student)
#   W0  stepping-in-place state, no hold (immediate switch)
#   X   W + everything of the report: start +-5 cm / 10 deg, contact / actuator / payload randomisation,
#       mocap velocity estimate, 0-40 ms latency, heading bias 3 deg, height bias 2 cm
#   XR  R + the same deviations
#   RP  rope start: set down in the frame-0 pose tilted +-3 deg, rope pull 30 % body weight released over 1 s, clock held
#       2.5 s in total, then the motion;  RPX the same + all deviations of X
#   RP10 slow rope release (user at least 10 s): pull 80 % body weight released over 10 s, clock held 12 s;
#        RP10X the same + all deviations
#   XS  X + stakes misplaced (every stake +-3 cm, whole layout 3 cm) + 2 cm obstacle perception noise
# Acceptance (fixed before the results,  request): D1, D2 >= 85 % in R and W (X reported), other
# layouts not more than 3 points below the model in A.
# usage: bash tools/rtF_accept.sh [export ...]   out robot_sim_check/out_rtF/<model>/, summary.md
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_rtF
export PYTHONPATH=$H/code HNU_STEP_STATES=$H/data/step_states/step_states_realF.npz
cd $H/code
declare -A EX=([mE]=$H/deploy_exports/v6g10xm_d2_s5_E/L3 [c2000]=$H/deploy_exports/rtF_s5_c2000/L3
               [c3000]=$H/deploy_exports/rtF_s5_c3000/L3 [r2000]=$H/deploy_exports/rtFr_s5_c2000/L3
               [s1500]=$H/deploy_exports/rtFs_s5_c1500/L3)
ALL="--start-xy 0.05 --start-yaw 10 --dr-x --vel-est --head-bias 3 --height-bias 0.02"
for M in ${@:-mE c2000 c3000}; do
  E=${EX[$M]}; [ -f $E/policy.onnx ] || continue
  mkdir -p $O/$M
  for L in $RS/v6/in/planned_layouts/*/; do
    L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
    for c in A R R3 RP RP10 W W0 X XR XS RPX RP10X RP10XS; do
      case $c in
        A) X="--latency 0-20";; R) X="--latency 0-20 --start rest --hold 1";; W) X="--latency 0-20 --start step --hold 1";;
        W0) X="--latency 0-20 --start step";; R3) X="--latency 0-20 --start rest --hold 3";; X) X="--latency 0-40 --start step --hold 1 $ALL";;
        XR) X="--latency 0-40 --start rest --hold 1 $ALL";;
        RP) X="--latency 0-20 --start rest --hold 2.5 --rope 0.3,1.0 --start-rp 3";;
        RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
        RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALL";;
        RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALL --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02";;
        RPX) X="--latency 0-40 --start rest --hold 2.5 --rope 0.3,1.0 --start-rp 3 $ALL";;
        XS) X="--latency 0-40 --start step --hold 1 $ALL --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02";;
      esac
      [ -f $O/$M/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 64 --tasks $t \
        --motion-dirs $L --latent-dir $L $X --out $O/$M/${n}_$c.json > $O/$M/${n}_$c.log 2>&1
    done
  done
done
$PYD - $O << 'EOF'
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
C = ["A", "R", "R3", "RP", "RP10", "W", "W0", "X", "XR", "XS", "RPX", "RP10X", "RP10XS"]
lines = ["# sim2sim acceptance (success v2 %, 64 trials; falls in brackets)", "",
         "A ref start | R rest + 1 s hold | W stepping + 1 s hold | W0 stepping, no hold | X = W + all deviations | XR = R + all", ""]
for M in sorted(os.listdir(O)):
    if not os.path.isdir(f"{O}/{M}"):
        continue
    lines += [f"## {M}", "", "| layout | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    tot = {c: [] for c in C}
    for n in sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{O}/{M}/*.json")}):
        row = []
        for c in C:
            f = f"{O}/{M}/{n}_{c}.json"
            if not os.path.exists(f):
                row.append("-"); continue
            R = json.load(open(f))["rows"]
            s = 100 * np.mean([r["success_v2"] for r in R]); fl = sum(r["cause"] == "fall" for r in R)
            tot[c].append(s); row.append(f"{s:.0f} ({fl})")
        lines.append(f"| {n.split('_')[0]} | " + " | ".join(row) + " |")
    lines.append("| mean | " + " | ".join(f"{np.mean(tot[c]):.0f}" if tot[c] else "-" for c in C) + " |")
    lines.append("")
open(f"{O}/summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
EOF
