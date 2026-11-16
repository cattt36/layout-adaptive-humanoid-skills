#!/bin/bash
# DEMO new G1: no-rope start check (the robot is NOT held by a rope on site) for exported 5PR candidates.
#   REPn (nominal) / BASEn (+ friction / CoM / encoder bias) / DRXn (actuator / contact DR) / REALn (= site: 2 cm / 3 deg
#   start, heading compensated with 1 deg error, 0-40 ms lag, base + actuator DR), standing start, no rope
#   OD=bh bash tools/rtJ_norope.sh <tag> <export_dir>  |  summary
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/${OD:-bh}
cd $H/code
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_PERCEPTION=1 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8
T=FiveStepPoleRun; ST="--start rest --hold 12 --start-rp 3"
C=(REPn BASEn DRXn REALn)
if [ "$1" = summary ]; then
  $PYD - $O "${C[@]}" << 'PEOF'
import ast, json, os, sys
import numpy as np
O, C = sys.argv[1], sys.argv[2:]
print("no rope: success v2 % (stake 1 stepped on %)\n")
print("| run | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
for t in os.environ.get("TAGS", "old").split():
    cells = []
    for k in C:
        f = f"{O}/{t}_{k}.json"
        if not os.path.exists(f):
            cells.append("-")
            continue
        r = json.load(open(f))["rows"]
        st = [x["stakes"] if not isinstance(x["stakes"], str) else ast.literal_eval(x["stakes"]) for x in r]
        s1 = np.mean([(s[0]["on"] or 0) if s else 0 for s in st])
        cells.append(f"{100 * np.mean([bool(x['success_v2']) for x in r]):.0f} ({100 * s1:.0f})")
    print(f"| {t} | " + " | ".join(cells) + " |")
PEOF
  exit 0
fi
t=$1; E=$2; L=$O/lay_$t
run() {
  local k=$1 NT=$2; shift 2
  $PYD deploy/sim2sim.py $E --crit task --trials $NT --workers 16 --tasks $T --motion-dirs $L --latent-dir $L "$@" \
    --out $O/${t}_$k.json > $O/${t}_$k.log 2>&1
}
run REPn 128 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --no-base-rand &
run BASEn 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 &
run DRXn 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --dr-x &
run REALn 64 $ST --start-xy 0.02 --start-yaw 3 --yaw-comp --yaw-meas-noise 1 --latency 0-40 --dr-x &
wait
echo "[$(date "+%m-%d %H:%M:%S")] $t no-rope eval done"
