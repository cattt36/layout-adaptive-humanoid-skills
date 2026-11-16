#!/bin/bash
# E2E demonstration videos: tools/e2e_demo.py prep -> record the execution of every accepted demo with the
# chosen candidate's planner tables (data/z_cand_plan_f1/<slot>/Z3, geometry data/geom_cand_f1/<slot>) -> compose.
# Output: runs/videos/e2e_demo/<demo id>.mp4
H=/root/gpufree-data/hnu_v2; D=$H/data; PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_DR=0 HNU_START_FRAC=0 HNU_SPLIT_JSON=$D/clip_split_cv/fold1/split.json MUJOCO_GL=egl HNU_DISPLAY_RENAME=1
O=$H/runs/videos/e2e_demo
CUDA_VISIBLE_DEVICES= PYTHONPATH=$D/data_code $PY tools/e2e_demo.py prep || exit 1
$PY - <<'EOF' > /tmp/e2e_demo_jobs.txt
import json
import os
ids = os.environ.get("HNU_DEMO_IDS", "")
for d in json.load(open("/root/gpufree-data/hnu_v2/runs/videos/e2e_demo/demo.json", encoding="utf-8")):
    if d.get("candidates") and (not ids or d["id"] in ids.split(",")):
        print(d["id"], d["layout"], d["candidates"][d["chosen"]]["slot"])
EOF
while read ID L K; do
  GK=$D/geom_cand_f1/$K
  HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 CUDA_VISIBLE_DEVICES=0 $PY tools/record_batch.py --clips $L \
    --student "NEW_L3=$H/runs/students_v6g10xm_d2_s1/Z3/student_2000.pt:$D/z_cand_plan_f1/$K/Z3" --slow 1.5 --out $O/exec_$ID
done < /tmp/e2e_demo_jobs.txt
$PY tools/e2e_demo.py compose
