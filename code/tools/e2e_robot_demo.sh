#!/bin/bash
# E2E videos of the 9 real-robot layouts with the current deployment (tools/e2e_robot_demo.py).
# Out: runs/videos/e2e_robot_D/<id>_<layout>.mp4 (+ demo.json, exec_<id>.mp4, cand/<id>/<clip>/preview.json)
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
O=$H/runs/videos/e2e_robot_D; E=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
CUDA_VISIBLE_DEVICES= PYTHONPATH=$RS/pylibs_sk:$RS/pylibs_tf:/root/gpufree-data/hnu_v2_4090/data/data_code $PY tools/e2e_robot_demo.py prep
$PYD tools/e2e_robot_demo.py cands
for C in $O/cand/*/*/; do
  C=${C%/}; t=$(ls $C | grep _motion.npz | cut -d_ -f1)
  # preview: 4 nominal rollouts (trials 0-3); execution: trial 4 of the same setting with the state trace
  [ -f $C/preview.json ] || PYTHONPATH=$H/code $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 4 --workers 4 \
    --tasks $t --motion-dirs $C --latent-dir $C --out $C/preview.json > $C/preview.log 2>&1 &
done
wait
$PYD tools/e2e_robot_demo.py choose
$PYD - << 'EOF' > /tmp/e2e_robot_exec.txt
import json
for d in json.load(open("/root/gpufree-data/hnu_v2/runs/videos/e2e_robot_D/demo.json"))["demos"]:
    if "chosen" in d:
        print(d["id"], d["candidates"][d["chosen"]]["clip"], d["clip"].split("_")[0])
EOF
while read ID C T; do
  D=$O/cand/$ID/$C; mkdir -p $D/trace
  [ -f $D/trace/done ] || { HNU_S2S_TRACE=$D/trace PYTHONPATH=$H/code $PYD deploy/sim2sim.py $E --latency 0-20 --crit task \
    --trials 5 --workers 5 --tasks $T --motion-dirs $D --latent-dir $D --out $D/exec.json > $D/exec.log 2>&1 && touch $D/trace/done; } &
done < /tmp/e2e_robot_exec.txt
wait
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 $PY tools/e2e_robot_demo.py render
$PY tools/e2e_robot_demo.py compose
echo ALLDONE
