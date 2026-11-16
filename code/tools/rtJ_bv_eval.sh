#!/bin/bash
# DEMO: export + sim2sim of a fixed-stakes faulty-waist tracker rtJbv_<tag> (tools/rtJ_bv.sh) on ITS training
# layout (012 / 014 / 018A: human pattern, 018B: vK option B):
#   W0 / Wp30 / Wrp = waist stuck at (roll, pitch) (0, 0) / (0, +0.30) / (+0.30, +0.40) rad (REP), DEMO / y6c at Wp30,
#   K28 / K150 = waist NOT locked, PD to target 0 with the policy gains / kp 150 kd 4 (REP)
#   bash tools/rtJ_bv_eval.sh <task> <clip> <tag>     (env G = gpu for the export)
#   bash tools/rtJ_bv_eval.sh summary                 -> markdown table (success v2 = every stake stepped on + no fall)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/${OD:-bv}; mkdir -p $O
cd $H/code
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_S2S_ROPE_LEN=2.0 HNU_PERCEPTION=1 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8
REP="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand --start-yaw 1"
DEMO="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.02 --start-yaw 3 --dr-x"
Y6C="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.01 --no-base-rand --start-yaw 6 --yaw-comp --yaw-meas-noise 1"
C=(W0 Wp30 Wrp DEMO y6c K28 K150)
if [ "$1" = summary ]; then
  $PYD - $O "${C[@]}" << 'PEOF'
import ast, json, os, sys
import numpy as np
O, C = sys.argv[1], sys.argv[2:]
print("success v2 % = every stake stepped on + no fall (falls %); landings = mean offset along / lat (cm) per stake in Wp30\n")
print("| run | " + " | ".join(C) + " | landings |\n|---|" + "---|" * (len(C) + 1))
for t in os.environ.get("TAGS", "012 014 018A 018B 018Aw").split():
    cells = []
    for k in C:
        f = f"{O}/{t}_{k}.json"
        if not os.path.exists(f):
            cells.append("-")
            continue
        r = json.load(open(f))["rows"]
        cells.append(f"{100 * np.mean([bool(x['success_v2']) for x in r]):.0f} ({100 * np.mean([x['cause'] == 'fall' for x in r]):.0f})")
    ex = "-"
    f = f"{O}/{t}_Wp30.json"
    if os.path.exists(f):
        st = [x["stakes"] if not isinstance(x["stakes"], str) else ast.literal_eval(x["stakes"]) for x in json.load(open(f))["rows"] if x.get("stakes")]
        n = max(len(s) for s in st)
        m = lambda i, k: np.mean([s[i][k] for s in st if len(s) > i and s[i][k] is not None])  # noqa: E731
        ex = "; ".join(f"{i + 1}: {100 * m(i, 'along'):+.0f} / {100 * m(i, 'lat'):+.0f}" for i in range(n))
    print(f"| {t} | " + " | ".join(cells) + f" | {ex} |")
PEOF
  exit 0
fi
T=$1; c=$2; t=$3; G=${G:-4}; RUN=${RP:-rtJbv}_$t; E=$H/deploy_exports/${RP:-rtJbv}/$RUN; mkdir -p $E
case $t in
  *012|*014) SR=$S/ref_stakes_s93_h10;; *018A|*018Aw|*018Aj0|*018Ap) SR=$S/ref_stakes_s83_h10;; *018B) SR=$D/std_bvB;;
esac
L=$O/lay_$t; mkdir -p $L
cp $SR/$T/${T}_${c}_motion.npz $SR/$T/${T}_${c}_stakes.json $L/; cp $S/z_tables_cv_f1/Z3/${T}_${c}_z.npy $L/
CK=$(last $T $RUN)
[ -f $E/policy.onnx ] || env HNU_STAKES_ROOT=$SR CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $CK $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
$PYD tools/tracker_parity.py $E > /dev/null 2>&1
env HNU_STAKES_ROOT=$SR HNU_ACTOR_BLIND=1 HNU_WAIST_LOCK=1 HNU_TRACKER_CKPT=$CK HNU_CLIPS_ALLOW=$S/lists/bw_${T}_$c.txt CUDA_VISIBLE_DEVICES=$G \
  $PY tools/tracker_obs_check.py HNU-Track-$T-Stakes ${T}_$c $E $SR/$T 2>&1 | grep -a -E "max \||OBS CHECK" > $E/obs_check.txt
echo "[$(date '+%m-%d %H:%M:%S')] $RUN $CK $(tail -n1 $E/parity.txt) / $(tail -n1 $E/obs_check.txt) md5 $(md5sum < $E/policy.onnx | cut -c1-8)"
run() {  # tag trials waist-off(or -) sim2sim args...
  local k=$1 NT=$2 W=$3; shift 3
  local WL="--waist-lock"; [ "$W" = - ] && { WL=""; W=0,0; }
  [ $k = K150 ] && export HNU_S2S_WAIST_KP=150,4
  HNU_S2S_WAIST_OFF=$W $PYD deploy/sim2sim.py $E --crit task --trials $NT --workers 16 --tasks $T \
    --motion-dirs $L --latent-dir $L $WL "$@" --out $O/${t}_$k.json > $O/${t}_$k.log 2>&1
}
run W0 128 0,0 $REP & run Wp30 128 0,0.30 $REP & run Wrp 64 0.30,0.40 $REP & run DEMO 64 0,0.30 $DEMO &
wait
run y6c 64 0,0.30 $Y6C & run K28 64 - $REP & run K150 64 - $REP &
wait
echo "[$(date '+%m-%d %H:%M:%S')] $RUN eval: $($PYD -c "
import json, sys
for k in sys.argv[2:]:
    r = json.load(open(f'$O/${t}_{k}.json'))['rows']
    print(f'{k} {100 * sum(bool(x[\"success_v2\"]) for x in r) / len(r):.0f}', end='  ')
" x ${C[@]})"
