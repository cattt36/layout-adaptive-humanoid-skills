#!/bin/bash
# Teacher check of the candidate augmentations (tools/make_aug_variants.py; train trials only): teacher bank, and for
# information the v6g10m seed-1 L0 / L4 students, nominal + randomised (teacher) / nominal (students), task criterion.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
$PY tools/make_aug_variants.py || exit 1
export HF_HUB_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_START_FRAC=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
gpu() { G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0; }
for VAR in mirror appr085 appr115 dwell2; do
  R=$H/data/augchk/$VAR O=$H/runs/aug_check/$VAR
  mkdir -p $O
  export HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift
  for DRV in 0 1; do
    gpu; HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
      --latent-dir $R/z/Z4 --per-clip 16 --out $O/dr$DRV --tag _teacher > $O/teacher$DRV.log 2>&1 || tail -5 $O/teacher$DRV.log
  done
  for P in Z0 Z4; do
    gpu; HNU_DR=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10m_d2_s1/$P/student_2000.pt \
      --latent-dir $R/z/$P --per-clip 16 --out $O/dr0 --tag _$P > $O/$P.log 2>&1 || tail -5 $O/$P.log
  done
done
$PY - << 'EOF'
import json, os
H = "/root/gpufree-data/hnu_v2/runs"
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
print("success % on the augmented TRAIN clips, DuMu / ThreeStepPoleRun / FiveStepPoleRun (ThreeStepPoleRun / FiveStepPoleRun stakes all hit %)")
print("baseline (1.0x train trials) teacher nominal: see results_v6g10m_bank_teacher_dr0 (train 92/98/98)")
for var in ("mirror", "appr085", "appr115", "dwell2"):
    for who, dr in (("teacher", "0"), ("teacher", "1"), ("Z0", "0"), ("Z4", "0")):
        f = f"{H}/aug_check/{var}/dr{dr}/" + ("eval_teacher_teacher.json" if who == "teacher" else f"eval_student_{who}.json")
        if not os.path.exists(f):
            print(f"{var:8s} {who} dr{dr}: missing"); continue
        s = json.load(open(f))["summary"]
        succ = "/".join(f"{100 * s[f'{t}/all']['success']:.0f}" for t in T)
        hit = "/".join(f"{100 * (s[f'{t}/all'].get('stakes_all_hit') or 0):.0f}" for t in T[1:])
        name = {"Z0": "L0 m s1", "Z4": "L4 m s1"}.get(who, who)
        print(f"{var:8s} {name:8s} {'nominal' if dr == '0' else 'random '}: {succ:12s} ({hit})")
EOF
