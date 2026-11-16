#!/bin/bash
# A: re-evaluate v6g10t seed-1 L0 / L4 and the teacher bank with the corrected fall thresholds (task criterion).
# B: can the TEACHER follow moved stakes?  Teacher and the two students on stakes perturbed per stake by +-5 / +-8 cm
#    (HNU_STAKE_JITTER), nominal, task criterion; train trials are reported (no test-set decision).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task
CK=$(cat $N/v6g10t_teachers.txt)
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
# A
for DRV in 0 1; do for X in "" "--push"; do
  HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 $X --out $H/runs/results_v6g10t_bank_teacher_dr$DRV --tag _teacher > /dev/null 2>&1
  for P in Z0 Z4; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10t_d2_s1/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5t/$P --per-clip 32 $X --out $H/runs/results_v6g10t_s1_dr$DRV --tag _$P > /dev/null 2>&1
  done
done; done
echo "A done"
# B
O=$H/runs/jit_check
for J in 0 0.05 0.08; do
  HNU_STAKE_JITTER=$J HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom \
    --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 --out $O/j$J --tag _teacher > /dev/null 2>&1
  for P in Z0 Z4; do
    HNU_STAKE_JITTER=$J HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10t_d2_s1/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5t/$P --per-clip 16 --out $O/j$J --tag _$P > /dev/null 2>&1
  done
done
$PY - << 'EOF'
import json, os
O = "/root/gpufree-data/hnu_v2/runs/jit_check"
for who, f in (("teacher", "eval_teacher_teacher.json"), ("L0", "eval_student_Z0.json"), ("L4", "eval_student_Z4.json")):
    for J in ("0", "0.05", "0.08"):
        p = f"{O}/j{J}/{f}"
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["summary"]
        cells = [f"{t[:6]} succ {100 * s[f'{t}/train']['success']:.0f}% hit {100 * (s[f'{t}/train'].get('stakes_all_hit') or 0):.0f}% "
                 f"dist {100 * (s[f'{t}/train'].get('stake_centre_dist_m') or 0):.1f}cm" for t in ("ThreeStepPoleRun", "FiveStepPoleRun")]
        print(f"{who:8s} jitter +-{J:4s} m (train trials) | " + " | ".join(cells))
EOF
