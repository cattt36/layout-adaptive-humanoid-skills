#!/bin/bash
# 64-trial confirmation (RP10X / RP10XS as out_rtJ) of the best std-planned candidates per layout
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check; PYD=/root/gpufree-data/venvs/deploy/bin/python
cd $H/code
export PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
export HNU_S2S_ROPE_LEN=2.0 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03
export HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
O=$RS/out_std_confirm; mkdir -p $O
SEL="D1:CrouchedTraverse_028,CrouchedTraverse_029 D2:CrouchedTraverse_005,CrouchedTraverse_006 N3:CrouchedTraverse_005,CrouchedTraverse_006
S1:ThreeStepPoleRun_014,ThreeStepPoleRun_015 S2:ThreeStepPoleRun_001,ThreeStepPoleRun_014 N1:ThreeStepPoleRun_019,ThreeStepPoleRun_014
T1:FiveStepPoleRun_018,FiveStepPoleRun_020 T2:FiveStepPoleRun_015,FiveStepPoleRun_017 N2:FiveStepPoleRun_017,FiveStepPoleRun_020"
for s in $SEL; do
  id=${s%%:*}; L=$(ls -d $RS/std/cand_plans/${id}_*); t=$(basename $L | cut -d_ -f2)
  if [ $t = CrouchedTraverse ]; then M=rtJb_c2000; VE=""; else M=rtJs_c2000; VE="--vel-est"; fi
  for c in $(echo ${s#*:} | tr ',' ' '); do
    for cond in RP10X RP10XS; do
      X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.05 --start-yaw 10 --dr-x $VE --head-bias 3 --height-bias 0.02"
      [ $cond = RP10XS ] && X="$X --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"
      [ -f $O/${id}_${c}_$cond.json ] || $PYD deploy/sim2sim.py $H/deploy_exports/$M/L3 --crit task --trials 64 --workers 100 --tasks $t \
        --motion-dirs $L/$c --latent-dir $L/$c $X --out $O/${id}_${c}_$cond.json > /dev/null 2>&1
    done
  done
done
$PYD - <<'PY' > $O/summary.md
import glob, json, os
import numpy as np
O = "/root/gpufree-data/robot_sim_check/out_std_confirm"
print("| layout | candidate | RP10X | RP10XS | hold drift (median / p90) | falls while standing |\n|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{O}/*_RP10X.json")):
    b = os.path.basename(f)[:-len("_RP10X.json")]
    L, c = b.split("_", 1)
    r1, r2 = (json.load(open(x))["rows"] for x in (f, f.replace("_RP10X.json", "_RP10XS.json")))
    h = [x["hold_drift"][0] for x in r1 + r2 if x.get("hold_drift")]
    print(f"| {L} | {c} | {100 * np.mean([x['success_v2'] for x in r1]):.0f} | {100 * np.mean([x['success_v2'] for x in r2]):.0f} | "
          f"{100 * np.median(h):.0f} / {100 * np.percentile(h, 90):.0f} cm | {len(r1) + len(r2) - len(h)} |")
PY
echo done > $O/DONE
