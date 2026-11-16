#!/bin/bash
# DEMO: why does the vK 3PR (012) work on the new G1 but 5PR (018) not?  Same eval, torque log, both clips:
#   REPn (nominal, no rope) / REALn (site: 2 cm / 3 deg compensated start, 0-40 ms lag, base + actuator DR, no rope)
#   + reference step geometry (stake to stake displacement, flight time) -> bmJ/prcmp
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/prcmp; mkdir -p $O
cd $H/code
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_PERCEPTION=1 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_TORQUE=1
ST="--start rest --hold 12 --start-rp 3"
run() {  # tag export layout task cond trials args...
  local t=$1 E=$2 L=$3 T=$4 k=$5 NT=$6; shift 6
  $PYD deploy/sim2sim.py $E --crit task --trials $NT --workers 16 --tasks $T --motion-dirs $L --latent-dir $L "$@" \
    --out $O/${t}_$k.json > $O/${t}_$k.log 2>&1
}
E3=$H/deploy_exports/rtJbx/rtJbx_012; L3=$B/stk_final/012r
E5=$H/deploy_exports/rtJbx/rtJbx_018; L5=$B/stk_final/018B
for p in "3PR $E3 $L3 ThreeStepPoleRun" "5PR $E5 $L5 FiveStepPoleRun"; do
  set -- $p
  run $1 $2 $3 $4 REPn 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --no-base-rand &
  run $1 $2 $3 $4 REALn 64 $ST --start-xy 0.02 --start-yaw 3 --yaw-comp --yaw-meas-noise 1 --latency 0-40 --dr-x &
done
wait
$PYD - $O $L3/ThreeStepPoleRun_012 $L5/FiveStepPoleRun_018 << 'PEOF'
import ast, json, sys
import numpy as np
O = sys.argv[1]
print("== reference geometry (stake to stake, m; human stance on each stake)")
for name, pre in zip(("3PR", "5PR"), sys.argv[2:]):
    j = json.load(open(pre + "_stakes.json"))
    s = j["stakes"]
    xy = np.array([[q["x"], q["y"]] for q in s])
    d = np.diff(xy, axis=0)
    gap = [b["stance"][0] - a["stance"][1] for a, b in zip(s[:-1], s[1:])]
    print(name, {k: v for k, v in j.items() if k != "stakes"}, "feet", "".join(q["foot"] for q in s))
    print(name, "  stake xy", np.round(xy, 2).tolist())
    print(name, "  step dist", np.round(np.linalg.norm(d, axis=1), 2).tolist(), " dx", np.round(d[:, 0], 2).tolist(),
          " dy", np.round(d[:, 1], 2).tolist(), " stance-to-stance gap (frames)", gap,
          " stance len", [q["stance"][1] - q["stance"][0] for q in s])
for name in ("3PR", "5PR"):
    for k in ("REPn", "REALn"):
        r = json.load(open(f"{O}/{name}_{k}.json"))["rows"]
        st = [x["stakes"] if not isinstance(x["stakes"], str) else ast.literal_eval(x["stakes"]) for x in r]
        on = np.array([[(q["on"] or 0) for q in s] for s in st if s])
        tq = [x["torque"] for x in r if x.get("torque")]
        js = [j for j in tq[0] if j != "eff"]
        sat = {j: np.mean([q[j][2] for q in tq]) for j in js}
        p95 = {j: np.mean([q[j][1] for q in tq]) for j in js}
        top = sorted(sat, key=lambda j: -sat[j])[:5]
        print(f"\n== {name} {k}: success v2 {100 * np.mean([bool(x['success_v2']) for x in r]):.0f} %, "
              f"stake on % {np.round(100 * on.mean(0)).astype(int).tolist()}, causes "
              f"{dict(zip(*np.unique([str(x['cause']) for x in r], return_counts=True)))}")
        print("   share of steps at the effort limit (p95 |tau| / limit):",
              ", ".join(f"{j.replace('_joint', '')} {100 * sat[j]:.1f}% ({p95[j]:.0f}/{tq[0]['eff'].get(j, 0):.0f})" for j in top))
PEOF
