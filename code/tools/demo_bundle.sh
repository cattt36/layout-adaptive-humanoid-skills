#!/bin/bash
# DEMO ONLY: stage the leadership-demo bundle (blind replay trackers rtJbx*) on the server:
#   models/<name>/ policy.onnx(+.data), policy_meta.json, parity.txt, obs_check.txt  (scene.mjb NOT copied: one shared
#                  ../scene.mjb = the Deployment_mocap scene, md5 96fbf4aa..., added locally)
#   motions/<name>_motion.npz   props/<name>.md (prop coordinates, start frame) + <name>.png (placement figure)
#   sim/summary.md (acceptance) + sim/<name>_REP.md / _DEMO.md (repeatability)
# Usage: bash tools/demo_bundle.sh [stage_dir]   (after "bx spread done")
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; S=$D/std; B=/root/gpufree-data/robot_sim_check/bmJ
O=${1:-/root/gpufree-data/Deployment_J_demo_stage}
cd $H/code
rm -rf $O; mkdir -p $O/models $O/motions $O/props $O/sim $O/code
# flat trackers only (the stakes-on trackers rtJbx_* reached 0-6 % with randomisation -> not shipped)
declare -A NAME=([rtJbxf_005]=bridge_005_flat [rtJbxf_013]=bridge_013_flat [rtJbxf_014]=3step_014_flat
  [rtJbxf_012]=3step_012_flat [rtJbxf_018]=5step_018_flat [rtJbxf_016]=5step_016_flat)
# suggested two-post bridges (tools/beam_plan.py, lowest underside among the lines the robot passes under, >= 1.0 m)
declare -A BEAM=([rtJbxf_005]=5.44,-0.56,2.50,0.27,1.11 [rtJbxf_013]=4.69,0.51,1.20,2.14,1.19)
for RUN in "${!NAME[@]}"; do
  E=$H/deploy_exports/rtJbx/$RUN; n=${NAME[$RUN]}; C=${RUN##*_}
  [ -f $E/policy.onnx ] || { echo "missing $E"; continue; }
  T=$($PYD -c "import json; print(json.load(open('$E/policy_meta.json'))['tracker']['task'])")
  mkdir -p $O/models/$n
  cp $E/policy.onnx $E/policy_meta.json $E/parity.txt $E/obs_check.txt $O/models/$n/ 2>/dev/null
  [ -f $E/policy.onnx.data ] && cp $E/policy.onnx.data $O/models/$n/
  if [[ $RUN == *bxf_* ]]; then L=$B/flat/${T}_$C; else L=$B/lay/${T}_$C; fi
  cp $L/${T}_${C}_motion.npz $O/motions/${n}_motion.npz
  PJ=$(ls $B/lay/${T}_$C/*_stakes.json $B/lay/${T}_$C/*_bridge.json 2>/dev/null | head -1)
  SP=$B/spread_bx/${RUN}_REP
  $PYD tools/props_table.py $L/${T}_${C}_motion.npz $SP $PJ > $O/props/$n.md 2>&1
  $PYD tools/prop_figure.py $SP $L/${T}_${C}_motion.npz $O/props/$n.png $PJ > /dev/null 2>&1
  cp $B/spread_bx/${RUN}_REP.md $O/sim/${n}_REP.md; cp $B/spread_bx/${RUN}_DEMO.md $O/sim/${n}_DEMO.md 2>/dev/null
  if [ $T = CrouchedTraverse ]; then cp $B/spread_bx/beam_free_$C.txt $O/props/${n}_beam_free.txt; cp $B/spread_bx/beam_plan_$C.txt $O/props/${n}_beam_plan.md
    DEMO_BEAM=${BEAM[$RUN]} $PYD tools/prop_figure.py $SP $L/${T}_${C}_motion.npz $O/props/${n}_bridge_suggest.png $PJ > /dev/null 2>&1; fi
  echo "$RUN -> $n ($T, $(md5sum < $E/policy.onnx | cut -c1-8))" >> $O/models/index.txt
done
cp $B/out_bx/summary.md $O/sim/summary.md
cp deploy/hnu_deploy.py $O/code/
ls -R $O | head -80; du -sh $O
