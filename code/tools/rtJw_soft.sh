#!/bin/bash
# evening DEPLOYMENT ONLY: extra acceptance of the faulty-waist students (tools/rtJw2.sh) with the waist NOT
# locked but on PD to target 0, as on the real robot  (kp 28.5: waist pitch at the 0.52 stop ~1 s into every skill).
#   K28 = policy gains (real), K150 / K300 = stiffer waist roll / pitch gains (deployment option),
#   K150E15 = kp 150 but the waist motor gives at most 15 Nm (maximum seen in tau_est)
# Waits for "rtJw done" in runs/night/rtF.log, CPU only.
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
IT_S=${IT_S:-2000}
while ! grep -q "rtJw done" $N/rtF.log; do sleep 60; done
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
OA=$RS/out_rtJw_soft; mkdir -p $OA
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
export HNU_S2S_ROPE_LEN=2.0 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID="$MID" HNU_BASE_END="$END"
log "rtJw soft-waist acceptance start"
for TAG in rtJwb rtJws; do
  for C in $IT_S 1000; do
    M=${TAG}_c$C; E=$H/deploy_exports/$M/L3; mkdir -p $OA/$M
    for L in $RS/std/planned_layouts/*/; do
      L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
      [ $TAG = rtJwb ] && [ $t != CrouchedTraverse ] && continue
      [ $TAG = rtJws ] && [ $t = CrouchedTraverse ] && continue
      VE=""; [ $TAG = rtJws ] && VE="--vel-est"
      ALLB="--start-xy 0.05 --start-yaw 10 --dr-x $VE --head-bias 3 --height-bias 0.02"
      SJ=""; [ $t != CrouchedTraverse ] && SJ="--stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"
      X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB $SJ"
      for v in K28 K150 K300 K150E15; do
        case $v in K28) K="";; K150) K="150,4";; K300) K="300,6";; K150E15) K="150,4,15";; esac
        if [ -n "$K" ]; then export HNU_S2S_WAIST_KP=$K; else unset HNU_S2S_WAIST_KP; fi
        [ -f $OA/$M/${n%%_*}_$v.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 \
          --tasks $t --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n%%_*}_$v.json > /dev/null 2>&1
      done
      unset HNU_S2S_WAIST_KP
    done
  done
done
$PYD - <<'PY' > $OA/summary.md
import json, os
import numpy as np
O = "/root/gpufree-data/robot_sim_check/out_rtJw_soft"
C = ["K28", "K150", "K300", "K150E15"]
print("waist NOT locked, PD to target 0: success v2 % (median of the max waist pitch, rad), 64 trials\n")
for M in sorted(os.listdir(O)):
    if not os.path.isdir(f"{O}/{M}"):
        continue
    print(f"## {M}\n\n| layout | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
    for n in ["D1", "D2", "N3", "S1", "S2", "N1", "T1", "T2", "N2"]:
        cells = []
        for c in C:
            f = f"{O}/{M}/{n}_{c}.json"
            if not os.path.exists(f):
                cells.append("-")
                continue
            r = json.load(open(f))["rows"]
            cells.append(f"{100 * np.mean([x['success_v2'] for x in r]):.0f} ({np.median([x.get('waist_pitch_max', 0) for x in r]):.2f})")
        if any(x != "-" for x in cells):
            print(f"| {n} | " + " | ".join(cells) + " |")
    print()
PY
log "rtJw soft done"
