#!/bin/bash
# DEMO: export + sim2sim of one faulty-waist tracker rtJbw_<clip> (tools/rtJ_bw.sh), same props as the
# vK bundle (bridge at the reference pose bmJ/bz_a/dir_<clip>; stakes bmJ/stk_final/012r 014r 018B):
#   W0 / Wp30 / Wrp = waist stuck at (roll, pitch) (0, 0) / (0, +0.30) / (+0.30, +0.40) rad (REP), DEMO / y6c at Wp30,
#   K28 / K150 = waist NOT locked, PD to target 0 with the policy gains / kp 150 kd 4 (REP),
#   OLD = the shipped waist-free vK model with the waist stuck at Wp30 (REP) for comparison.
#   bash tools/rtJ_bw_eval.sh <task> <clip>     (env G = gpu for the export)
#   bash tools/rtJ_bw_eval.sh summary           -> markdown table
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/bw; mkdir -p $O
cd $H/code
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_S2S_ROPE_LEN=2.0 HNU_PERCEPTION=1
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand --start-yaw 1"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
Y6C="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand --start-yaw 6 --yaw-comp --yaw-meas-noise 1"
if [ "$1" = summary ]; then
  $PYD - $O << 'PEOF'
import ast, json, os, sys
import numpy as np
O = sys.argv[1]
C = ["W0", "Wp30", "Wrp", "DEMO", "y6c", "K28", "K150", "OLD"]
print("success v2 % (falls %) per condition; bridge: + weaving kept (robot runs) in REP Wp30; stakes: mean landing offset "
      "along / lat (cm) per stake in REP Wp30\n")
print("| clip | " + " | ".join(C) + " | weaving / landings |\n|---|" + "---|" * (len(C) + 1))
for c in ("013", "005", "012", "014", "018"):
    cells = []
    for k in C:
        f = f"{O}/{c}_{k}.json"
        if not os.path.exists(f):
            cells.append("-")
            continue
        r = json.load(open(f))["rows"]
        cells.append(f"{100 * np.mean([x['success_v2'] for x in r]):.0f} ({100 * np.mean([x['cause'] == 'fall' for x in r]):.0f})")
    extra = "-"
    t = f"{O}/{c}_Wp30_topo.txt"
    if os.path.exists(t):
        w = [l.split(":", 1)[1].strip() for l in open(t) if l.startswith("robot runs")]
        extra = w[0] if w else "-"
    f = f"{O}/{c}_Wp30.json"
    if c in ("012", "014", "018") and os.path.exists(f):
        st = [x["stakes"] if not isinstance(x["stakes"], str) else ast.literal_eval(x["stakes"]) for x in json.load(open(f))["rows"] if x.get("stakes")]
        n = max(len(s) for s in st)
        extra = "; ".join(f"{i + 1}: {100 * np.mean([s[i]['along'] for s in st if len(s) > i and s[i]['along'] is not None]):+.0f} / "
                          f"{100 * np.mean([s[i]['lat'] for s in st if len(s) > i and s[i]['lat'] is not None]):+.0f}" for i in range(n))
    print(f"| {c} | " + " | ".join(cells) + f" | {extra} |")
PEOF
  exit 0
fi
T=$1; c=$2; G=${G:-4}; RUN=rtJbw_$c; E=$H/deploy_exports/rtJbw/$RUN; mkdir -p $E
if [ $T = CrouchedTraverse ]; then
  export HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID="$MID" HNU_BASE_END="$END"
  L=$B/bz_a/dir_$c; X="HNU_BRIDGE_ROOT=$D/std_ctreal"; MD=$D/std_ctreal/$T; SUF=-Bridge
  [ $c = 013 ] && OLD=$H/deploy_exports/rtJbz/rtJbz_013w10 || OLD=$H/deploy_exports/rtJbz/rtJbz_005j2
else
  export HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8
  case $c in 012) L=$B/stk_final/012r;; 014) L=$B/stk_final/014r;; 018) L=$B/stk_final/018B;; esac
  [ $T = ThreeStepPoleRun ] && SR=ref_stakes_s93_h10 || SR=ref_stakes_s83_h10
  X="HNU_STAKES_ROOT=$S/$SR"; MD=$S/$SR/$T; SUF=-Stakes; OLD=$H/deploy_exports/rtJbx/rtJbx_$c
fi
CK=$(last $T $RUN)
[ -f $E/policy.onnx ] || env $X CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $CK $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
$PYD tools/tracker_parity.py $E > /dev/null 2>&1
env $X HNU_ACTOR_BLIND=1 HNU_WAIST_LOCK=1 HNU_TRACKER_CKPT=$CK HNU_CLIPS_ALLOW=$S/lists/bw_${T}_$c.txt CUDA_VISIBLE_DEVICES=$G \
  $PY tools/tracker_obs_check.py HNU-Track-$T$SUF ${T}_$c $E $MD 2>&1 | grep -a -E "max \||OBS CHECK" > $E/obs_check.txt
echo "[$(date '+%m-%d %H:%M:%S')] $RUN $CK $(tail -n1 $E/parity.txt) / $(tail -n1 $E/obs_check.txt) md5 $(md5sum < $E/policy.onnx | cut -c1-8)"
run() {  # tag export trials waist-off(or -) sim2sim args...
  local k=$1 M=$2 NT=$3 W=$4; shift 4
  local WL="--waist-lock"; [ "$W" = - ] && { WL=""; W=0,0; }
  [ $k = K150 ] && export HNU_S2S_WAIST_KP=150,4
  local TD=""; [ $T = CrouchedTraverse ] && { TD=$O/trace_${c}_$k; rm -rf $TD; mkdir -p $TD; }
  HNU_S2S_TRACE=$TD HNU_S2S_WAIST_OFF=$W $PYD deploy/sim2sim.py $M --crit task --trials $NT --workers 24 --tasks $T \
    --motion-dirs $L --latent-dir $L $WL "$@" --out $O/${c}_$k.json > $O/${c}_$k.log 2>&1
  [ $T = CrouchedTraverse ] && $PYD tools/ct_topo.py $M $L/${T}_${c}_motion.npz $TD 12 $L/${T}_${c}_bridge.json > $O/${c}_${k}_topo.txt 2>&1
  [ $T = CrouchedTraverse ] && [ $k != Wp30 ] && rm -rf $TD
}
run W0 $E 128 0,0 $REP &
run Wp30 $E 128 0,0.30 $REP &
run Wrp $E 64 0.30,0.40 $REP &
run DEMO $E 64 0,0.30 $DEMO &
wait
run y6c $E 64 0,0.30 $Y6C &
run K28 $E 64 - $REP &
run K150 $E 64 - $REP &
run OLD $OLD 64 0,0.30 $REP &
wait
echo "[$(date '+%m-%d %H:%M:%S')] $RUN eval: $(for k in W0 Wp30 Wrp DEMO y6c K28 K150 OLD; do echo -n "$k $(grep -a -o 'success *[0-9.]*%' $O/${c}_$k.log | tail -1 | tr -s ' ' | cut -d' ' -f2) "; done)"
