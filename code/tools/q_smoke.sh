#!/bin/bash
# Smoke test of the execution-quality metrics in eval_student.py.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
rm -rf /tmp/q_smoke
CUDA_VISIBLE_DEVICES=1 timeout 1200 $PY eval_student.py --student $H/runs/students_v5_d2/Z3/student_2000.pt \
  --latent-dir $H/data/z_tables_v5/Z3 --per-clip 1 --out /tmp/q_smoke > /tmp/q_smoke.log 2>&1
grep -a -e Error -e Traceback /tmp/q_smoke.log | head -n 3
$PY - << 'EOF'
import json
s = json.load(open("/tmp/q_smoke/eval_student.json"))["summary"]
for k in ("CrouchedTraverse/all", "ThreeStepPoleRun/all"):
    print(k, s[k])
EOF
