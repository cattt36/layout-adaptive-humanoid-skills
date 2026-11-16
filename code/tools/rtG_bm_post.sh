#!/bin/bash
# after tools/rtG_bm.sh -- export the three blind trackers (tools/export_tracker.py + tracker_parity.py),
# runtime observation check in the training env (tools/tracker_obs_check.py), CPU sim2sim acceptance on the 9 planned
# layouts (the tracker of the layout's task plays the layout's reference motion.npz), 64 trials:
#   R3 rest + 3 s hold | RP10 rope 80 % BW released over 10 s, 12 s hold | RP10X + start 5 cm / 5 deg, contact /
#   actuator / payload randomisation, 0-40 ms | RP10XS + stakes misplaced 3 cm each and 3 cm as a whole
# Out: deploy_exports/$RUN/<task>/, robot_sim_check/out_$RUN/summary.md
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
RUN=${RUN:-rtGbm}
while ! grep -q "blind trackers $RUN done" $N/rtF.log; do sleep 60; done
BASE=$H/deploy_exports/rtFs_s5_c1500/L3
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ACTOR_BLIND=1 HNU_WAIST_LOCK=1 HNU_PERCEPTION=1 HNU_POST_BASE=1 \
  HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 HNU_BASE_MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}' \
  HNU_BASE_END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}' HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8
declare -A SUF=([CrouchedTraverse]=-Bridge [ThreeStepPoleRun]=-Stakes [FiveStepPoleRun]=-Stakes)
declare -A CLIP=([CrouchedTraverse]=CrouchedTraverse_024 [ThreeStepPoleRun]=ThreeStepPoleRun_011 [FiveStepPoleRun]=FiveStepPoleRun_019)
declare -A MDIR=([CrouchedTraverse]=$D/ref_bridge_video_lift/CrouchedTraverse [ThreeStepPoleRun]=$D/ref_stakes_s93_h10/ThreeStepPoleRun
                 [FiveStepPoleRun]=$D/ref_stakes_s83_h10/FiveStepPoleRun)
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  E=$H/deploy_exports/$RUN/$T; CK=$(last $T $RUN); mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=4 $PY tools/export_tracker.py $CK $T $BASE $E > $E.export.log 2>&1
  $PYD tools/tracker_parity.py $E > /dev/null 2>&1
  [ $T = ThreeStepPoleRun ] && SR=HNU_STAKES_ROOT=$D/ref_stakes_s93_h10 || SR=HNU_STAKES_ROOT=$D/ref_stakes_s83_h10
  env $SR HNU_TRACKER_CKPT=$CK CUDA_VISIBLE_DEVICES=4 $PY tools/tracker_obs_check.py HNU-Track-$T${SUF[$T]} ${CLIP[$T]} $E ${MDIR[$T]} \
    2>&1 | grep -a -E "max \||OBS CHECK" > $E/obs_check.txt
  log "tracker $T: $(tail -n 1 $E/parity.txt) / $(tail -n 1 $E/obs_check.txt)"
done
O=$RS/out_$RUN; mkdir -p $O
ALLB="--start-xy 0.05 --start-yaw 5 --dr-x"
for L in $RS/v6/in/planned_layouts/*/; do
  L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2); E=$H/deploy_exports/$RUN/$t
  for c in R3 RP10 RP10X RP10XS; do
    case $c in R3) X="--latency 0-20 --start rest --hold 3";;
      RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
      RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB";;
      RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03";; esac
    [ -f $O/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $t --waist-lock \
      --motion-dirs $L --latent-dir $L $X --out $O/${n}_$c.json > $O/${n}_$c.log 2>&1
  done
done
$PYD - $O << 'PEOF' > $O/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
C = ["R3", "RP10", "RP10X", "RP10XS"]
print("## blind trackers (64 trials, success v2 %; swing-scuff steps in RP10X)\n\n| layout | " + " | ".join(C) + " | scuff |\n|---|" + "---|" * (len(C) + 1))
for n in sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{O}/*.json")}):
    row = []
    for c in C:
        f = f"{O}/{n}_{c}.json"
        row.append(f"{100 * np.mean([r['success_v2'] for r in json.load(open(f))['rows']]):.0f}" if os.path.exists(f) else "-")
    f = f"{O}/{n}_RP10X.json"
    sc = np.mean([r.get("swing_scuff_steps", 0) for r in json.load(open(f))["rows"]]) if os.path.exists(f) else float("nan")
    print(f"| {n.split('_')[0]} | " + " | ".join(row) + f" | {sc:.1f} |")
PEOF
log "$RUN post done"
