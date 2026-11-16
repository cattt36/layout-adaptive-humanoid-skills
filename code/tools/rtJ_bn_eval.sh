#!/bin/bash
# DEMO new G1: export + sim2sim of rtJbn_<tag> (tools/rtJ_bn.sh) on its training layout, waist free.
#   REP (nominal) / BASE (+ friction / CoM / encoder bias) / L40 (0-40 ms lag) / DRX (actuator / contact DR) /
#   YAW3 (2 cm / 3 deg start error, no compensation) / DEMO (all of them) / y6c (6 deg, compensated +-1 deg)
#   bash tools/rtJ_bn_eval.sh <tag>  |  summary  (tag "old" = the vK model rtJbx_018 on layout 018B)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/${OD:-bn}; mkdir -p $O
cd $H/code
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_S2S_ROPE_LEN=2.0 HNU_PERCEPTION=1 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_EXPORT_WAIST_FREE=1
ST="--start rest --hold 12 --rope 0.8,10 --start-rp 3"
C=(REP BASE L40 DRX YAW3 DEMO y6c)
if [ "$1" = summary ]; then
  $PYD - $O "${C[@]}" << 'PEOF'
import ast, json, os, sys
import numpy as np
O, C = sys.argv[1], sys.argv[2:]
print("success v2 % = every stake stepped on + no fall (stake 1 stepped on %)\n")
print("| run | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
for t in os.environ.get("TAGS", "old B Bp A Bj6 By5").split():
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
t=$1; G=${G:-4}; T=FiveStepPoleRun
if [ $t = old ]; then
  E=$H/deploy_exports/rtJbx/rtJbx_018; SR=$D/std_bvB
else
  RUN=${RP:-rtJbn}_$t; E=$H/deploy_exports/${RP:-rtJbn}/$RUN; mkdir -p $E; CK=$(last $T $RUN)
  case $t in *A*) SR=$S/ref_stakes_s83_h10;; *) SR=$D/std_bvB;; esac
  [ -n "$SRO" ] && SR=$SRO  # training stakes root override (rtJ_bl.sh: warped layouts)
  [ -f $E/policy.onnx ] || env HNU_STAKES_ROOT=$SR CUDA_VISIBLE_DEVICES=$G $PY tools/export_tracker.py $CK $T $H/deploy_exports/rtJb_c2000/L3 $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  env HNU_STAKES_ROOT=$SR HNU_ACTOR_BLIND=1 HNU_TRACKER_CKPT=$CK HNU_CLIPS_ALLOW=$S/lists/bw_${T}_018.txt CUDA_VISIBLE_DEVICES=$G \
    $PY tools/tracker_obs_check.py HNU-Track-$T-Stakes ${T}_018 $E $SR/$T 2>&1 | grep -a -E "max \||OBS CHECK" > $E/obs_check.txt
  echo "[$(date '+%m-%d %H:%M:%S')] $RUN $CK $(tail -n1 $E/parity.txt) / $(tail -n1 $E/obs_check.txt) md5 $(md5sum < $E/policy.onnx | cut -c1-8)"
fi
L=$O/lay_$t; mkdir -p $L; rm -rf $O/tr_$t; mkdir -p $O/tr_$t
cp $SR/$T/${T}_018_motion.npz $SR/$T/${T}_018_stakes.json $L/; cp $S/z_tables_cv_f1/Z3/${T}_018_z.npy $L/
run() {
  local k=$1 NT=$2; shift 2
  local TR=""; [ $k = REP ] && TR=$O/tr_$t
  HNU_S2S_TRACE=$TR $PYD deploy/sim2sim.py $E --crit task --trials $NT --workers 16 --tasks $T --motion-dirs $L --latent-dir $L "$@" \
    --out $O/${t}_$k.json > $O/${t}_$k.log 2>&1
}
run REP 128 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --no-base-rand &
run BASE 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 &
run L40 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-40 --no-base-rand &
run DRX 64 $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --dr-x &
wait
run YAW3 64 $ST --start-xy 0.02 --start-yaw 3 --latency 0-20 --no-base-rand &
run DEMO 64 $ST --start-xy 0.02 --start-yaw 3 --latency 0-40 --dr-x &
run y6c 64 $ST --start-xy 0.01 --start-yaw 6 --yaw-comp --yaw-meas-noise 1 --latency 0-20 --no-base-rand &
wait
$PYD tools/hop_height.py $E $L/${T}_018_motion.npz $O/tr_$t > $O/${t}_hop.txt 2>&1; rm -rf $O/tr_$t
echo "[$(date "+%m-%d %H:%M:%S")] $t eval done: $(tail -n 2 $O/${t}_hop.txt | paste -sd' ')"
