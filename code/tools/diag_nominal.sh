#!/bin/bash
# Why are the v6g10 students worse in the nominal env (no randomisation, zero latency) than in the randomised env?
# Evaluation only (clean), seed-1 students Z0 / Z3, one factor at a time:
#   nom       HNU_DR=0                                      (already in results_v6g10_s1_dr0)
#   lag0      latency forced to 0, no dynamics DR            -> must equal nom (checks the delay-buffer path)
#   lag1      constant 20 ms, nominal dynamics
#   lag01     piecewise 0-20 ms, nominal dynamics
#   dyn       dynamics DR (PD / effort / mass), zero latency
#   full      dynamics DR + 0-20 ms                           (= dr1)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17  # same scene as run_v6g10.sh
G=${G:-1}
O=$H/runs/diag_nominal_v6g10; mkdir -p $O
for P in Z0 Z3; do
  ST=$H/runs/students_v6g10_d2_s1/$P/student_2000.pt
  for C in "lag0 lag 0 0" "lag1 lag 1 1" "lag01 lag 0 1" "dyn dyn 0 0" "full dyn,lag 0 1"; do
    set -- $C
    [ -f $O/$1/eval_student_$P.json ] && continue
    HNU_DR=1 HNU_DR_LEVEL=1 HNU_DR_PARTS=$2 HNU_DR_LAG_MIN=$3 HNU_DR_LAG_MAX=$4 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G \
      $PY eval_student.py --student $ST --latent-dir $H/data/z_tables_v5/$P --per-clip 8 --out $O/$1 --tag _$P >> $O/log.txt 2>&1
  done
done
$PY - <<'EOF'
import json, os
H = "/root/gpufree-data/hnu_v2/runs"
for T in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    for P in ("Z0", "Z3"):
        row = []
        for c, d in (("nom", f"{H}/results_v6g10_s1_dr0"), ("lag0", None), ("lag1", None), ("lag01", None), ("dyn", None),
                     ("full", None)):
            f = f"{d or H + '/diag_nominal_v6g10/' + c}/eval_student_{P}.json"
            row.append(f"{c} {100 * json.load(open(f))['summary'][T + '/all']['success']:.1f}" if os.path.exists(f) else f"{c} -")
        print(T, P, " | ".join(row))
EOF
