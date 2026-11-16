#!/bin/bash
# preview-like sim of every std-planned candidate (tools/std_plan.py) on the 9 standard-start layouts,
# new-robot students rtJb / rtJs c2000, real conditions (as out_rtJ RP10X), 16 trials -> out_std_cands
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check; PYD=/root/gpufree-data/venvs/deploy/bin/python
cd $H/code
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
export HNU_S2S_ROPE_LEN=2.0 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03
export HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
O=$RS/out_std_cands; mkdir -p $O
for L in $RS/std/cand_plans/*/; do
  L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
  if [ $t = CrouchedTraverse ]; then M=rtJb_c2000; VE=""; SJ=""; else M=rtJs_c2000; VE="--vel-est"; SJ="--stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"; fi
  for C in $L/*/; do
    C=${C%/}; c=$(basename $C)
    [ -f $O/${n%%_*}_$c.json ] || $PYD deploy/sim2sim.py $H/deploy_exports/$M/L3 --crit task --trials 16 --workers 100 --tasks $t \
      --motion-dirs $C --latent-dir $C --latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.05 \
      --start-yaw 10 --dr-x $VE --head-bias 3 --height-bias 0.02 $SJ --out $O/${n%%_*}_$c.json > /dev/null 2>&1
  done
done
$PYD - <<'PY' > $O/summary.md
import glob, json, os
import numpy as np
O = "/root/gpufree-data/robot_sim_check/out_std_cands"
rows = {}
for f in sorted(glob.glob(f"{O}/*.json")):
    L, c = os.path.basename(f)[:-5].split("_", 1)
    r = json.load(open(f))["rows"]
    h = [x["hold_drift"][0] for x in r if x.get("hold_drift")]
    rows.setdefault(L, []).append((c, 100 * np.mean([x["success_v2"] for x in r]), np.median(h) if h else float("nan"), len(r) - len(h)))
print("std-planned candidates, 16 trials each, real conditions: success v2 % / median hold drift / falls while standing\n")
for L, v in rows.items():
    print(f"## {L}\n" + "\n".join(f"- {c}: {s:.0f} %, drift {100 * d:.0f} cm" + (f", {k} fell standing" if k else "") for c, s, d, k in sorted(v, key=lambda x: -x[1])) + "\n")
PY
echo done > $O/DONE
