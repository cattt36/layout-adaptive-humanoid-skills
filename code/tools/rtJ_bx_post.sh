#!/bin/bash
# second attempt (tools/rtJ_bx.sh, drift limit 1.0 m): same post-processing as rtJ_bm_post.sh
# CT trackers are flat (rtJbxf_005/013, tools/rtJ_bxc.sh, data/std_flatct: bridge parked)
# DEMO ONLY: after tools/rtJ_bm.sh (bridge + stakes-on trackers) and tools/rtJ_bmf.sh (flat stakes
# trackers): export (waist free, base = rtJb c2000 = new-G1 scene), ONNX parity, runtime observation check, CPU sim2sim
# on the clip's own reference (props at the reference positions), 64 trials:
#   N    rest start, 3 s hold, 0-20 ms
#   RP   rope 80 % BW released over 10 s, 12 s hold, start roll / pitch 3 deg, 0-20 ms
#   RPX  RP + start 2 cm / 3 deg (floor template), contact / actuator / payload randomisation, pushes, 0-40 ms
#   RPXO RPX + props misplaced: stakes 5 cm each + 3 cm as a whole / bridge 5 cm as a whole   (not for flat)
# Out: deploy_exports/rtJbx/<run>/, robot_sim_check/bmJ/out/summary.md, log "bx post done"
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; N=$H/runs/night; B=/root/gpufree-data/robot_sim_check/bmJ
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
[ -n "$NOWAIT" ] || while ! grep -q "bx train done" $N/rtF.log || ! grep -q "bxc train done" $N/rtF.log; do sleep 60; done
BASE=$H/deploy_exports/rtJb_c2000/L3
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 \
  HNU_STAKE_TOL=0.17 HNU_ACTOR_BLIND=1 HNU_PERCEPTION=1 HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' \
  HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_ROPE_LEN=2.0 HNU_EXPORT_WAIST_FREE=1
# eval dirs: clip motion + props json (+ z table, only read for the clock length)
for c in CrouchedTraverse_005:ref_bridge_video_lift:bridge CrouchedTraverse_013:ref_bridge_video_lift:bridge ThreeStepPoleRun_014:ref_stakes_s93_h10:stakes \
         ThreeStepPoleRun_012:ref_stakes_s93_h10:stakes FiveStepPoleRun_018:ref_stakes_s83_h10:stakes FiveStepPoleRun_016:ref_stakes_s83_h10:stakes; do
  n=${c%%:*}; r=$(echo $c | cut -d: -f2); k=${c##*:}; t=${n%_*}; mkdir -p $B/lay/$n; cp $S/$r/$t/${n}_motion.npz $S/$r/$t/${n}_$k.json $B/lay/$n/; done
mkdir -p $B/flat
for c in ThreeStepPoleRun_014 ThreeStepPoleRun_012 FiveStepPoleRun_018 FiveStepPoleRun_016 CrouchedTraverse_005 CrouchedTraverse_013; do
  F=$D/std_flat; [ ${c%_*} = CrouchedTraverse ] && F=$D/std_flatct
  mkdir -p $B/flat/$c; cp $F/${c%_*}/${c}_motion.npz $B/flat/$c/; done
for d in $B/lay/* $B/flat/*; do c=$(basename $d); f=$(ls $S/z_tables_cv_f1/Z3/${c}_z.npy $S/z_tables_cv_f1_xzmix/Z3/${c}_z.npy 2>/dev/null | head -1)
  [ -n "$f" ] && cp $f $d/ || log "bm post: no z table for $c"; done
RUNS=${RUNS_OVERRIDE:-"CrouchedTraverse:005:rtJbxf CrouchedTraverse:013:rtJbxf ThreeStepPoleRun:014:rtJbx ThreeStepPoleRun:012:rtJbx
      FiveStepPoleRun:018:rtJbx FiveStepPoleRun:016:rtJbx ThreeStepPoleRun:014:rtJbxf ThreeStepPoleRun:012:rtJbxf
      FiveStepPoleRun:018:rtJbxf FiveStepPoleRun:016:rtJbxf"}
for r in $RUNS; do
  T=$(echo $r | cut -d: -f1); C=$(echo $r | cut -d: -f2); P=$(echo $r | cut -d: -f3); RUN=${P}_$C
  E=$H/deploy_exports/${EXP:-rtJbx}/$RUN; CK=$(last $T $RUN); mkdir -p $E
  SUF=-Stakes; [ $T = CrouchedTraverse ] && SUF=-Bridge
  if [[ $P == *bxf* ]] && [ $T = CrouchedTraverse ]; then MD=$D/std_flatct/$T; X="HNU_BRIDGE_ROOT=$D/std_flatct"
  elif [[ $P == *bxf* ]]; then MD=$D/std_flat/$T; X="HNU_STAKES_ROOT=$D/std_flat HNU_STAKE_CONTACT=0 HNU_RUNUP_FREE_S=-1"
  elif [ $T = CrouchedTraverse ]; then MD=$S/ref_bridge_video_lift/$T; X="HNU_BRIDGE_ROOT=$S/ref_bridge_video_lift"
  elif [ $T = ThreeStepPoleRun ]; then MD=$S/ref_stakes_s93_h10/$T; X="HNU_STAKES_ROOT=$S/ref_stakes_s93_h10"
  else MD=$S/ref_stakes_s83_h10/$T; X="HNU_STAKES_ROOT=$S/ref_stakes_s83_h10"; fi
  [ -f $E/policy.onnx ] || env $X CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $CK $T $BASE $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  env $X HNU_TRACKER_CKPT=$CK HNU_CLIPS_ALLOW=$S/lists/bm_${T}_$C.txt CUDA_VISIBLE_DEVICES=4 $PY tools/tracker_obs_check.py \
    HNU-Track-$T$SUF ${T}_$C $E $MD 2>&1 | grep -a -E "max \||OBS CHECK" > $E/obs_check.txt
  log "bm $RUN: $(tail -n 1 $E/parity.txt) / $(tail -n 1 $E/obs_check.txt)"
done
O=${OUT:-$B/out_bx}; mkdir -p $O
X0="--start-xy 0.02 --start-yaw 3 --dr-x --push"
for r in $RUNS; do
  T=$(echo $r | cut -d: -f1); C=$(echo $r | cut -d: -f2); P=$(echo $r | cut -d: -f3); RUN=${P}_$C; E=$H/deploy_exports/${EXP:-rtJbx}/$RUN
  if [[ $P == *bxf* ]]; then L=$B/flat/${T}_$C; CR=track; CONDS="N RP RPX"; SE="HNU_STAKES_ROOT=$D/std_flat"
  else L=$B/lay/${T}_$C; CR=task; CONDS="N RP RPX RPXO"; SE=""; fi
  for c in $CONDS; do
    case $c in N) A="--latency 0-20 --start rest --hold 3";;
      RP) A="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
      RPX) A="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $X0";;
      RPXO) [ $T = CrouchedTraverse ] && A="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $X0 --obst-shift 0.05" \
                                     || A="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $X0 --stake-jitter 0.05 --obst-shift 0.03";; esac
    [ -f $O/${RUN}_$c.json ] || env $SE $PYD deploy/sim2sim.py $E --crit $CR --trials ${TRIALS:-64} --workers 100 --tasks $T \
      --motion-dirs $L --latent-dir $L $A --out $O/${RUN}_$c.json > $O/${RUN}_$c.log 2>&1
  done
done
$PYD - $O << 'PEOF' > $O/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
C = ["N", "RP", "RPX", "RPXO"]
print("## demo trackers rtJbx (64 trials, % success; stakes-on / bridge: no fall + task, flat: no fall + tracked)\n")
print("| run | " + " | ".join(C) + " | RPX causes |\n|---|" + "---|" * (len(C) + 1))
for n in sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{O}/*.json")}):
    row = []
    for c in C:
        f = f"{O}/{n}_{c}.json"
        row.append(f"{100 * np.mean([r['success_v2'] if 'success_v2' in r else r['success'] for r in json.load(open(f))['rows']]):.0f}"
                   if os.path.exists(f) else "-")
    f = f"{O}/{n}_RPX.json"
    cs = {}
    if os.path.exists(f):
        for r in json.load(open(f))["rows"]:
            cs[r.get("cause", "?")] = cs.get(r.get("cause", "?"), 0) + 1
    print(f"| {n} | " + " | ".join(row) + " | " + ", ".join(f"{k} {v}" for k, v in sorted(cs.items())) + " |")
PEOF
log "bx post done"
