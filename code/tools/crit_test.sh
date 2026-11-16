#!/bin/bash
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
for C in track task; do
  HNU_EVAL_CRIT=$C HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=1 /root/gpufree-data/venvs/mjlab/bin/python eval_student.py \
    --student $H/runs/students_v6g10p_d2_s1/Z0A/student_1000.pt --latent-dir $H/data/z_tables_v5/Z0 --per-clip 4 \
    --out /tmp/crit_test_$C --tag _Z0A 2>&1 | grep -e "/all" -e "/val" -e Error -e Traceback | head -n 8
  /root/gpufree-data/venvs/mjlab/bin/python -c "import json; s=json.load(open('/tmp/crit_test_$C/eval_student_Z0A.json'))['summary']; print('$C', {k: (v['success'], v.get('stakes_all_hit'), v.get('stake_centre_dist_m')) for k, v in s.items() if k.endswith('/all')})"
done