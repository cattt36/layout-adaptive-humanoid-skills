#!/bin/bash
# Pilot of the pre-registered secondary experiments on v6g10m seed 1 (L0 / L4), 16 rollouts per clip, nominal:
#   B semantic consistency (HNU_EVAL_SEM=1)      C motion-capture dropout, obstacle features frozen 0.5 / 1.0 s
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
$PY tools/make_freeze_table.py || exit 1
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 \
  HNU_PERC_FREEZE_TABLE=$H/runs/freeze_table.json
O=$H/runs/sec_pilot
gpu() { G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0; }
for P in Z0 Z4; do
  gpu; HNU_EVAL_SEM=1 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10m_d2_s1/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5t/$P --per-clip 16 --out $O/sem --tag _$P > $O.sem_$P.log 2>&1 || tail -5 $O.sem_$P.log
  for F in 0.5 1.0; do
    gpu; HNU_PERC_FREEZE=$F CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10m_d2_s1/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5t/$P --per-clip 16 --out $O/frz$F --tag _$P > $O.frz${F}_$P.log 2>&1 || tail -5 $O.frz${F}_$P.log
  done
done
$PY - << 'EOF'
import json, os
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/sec_pilot"
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
for P in ("Z0", "Z4"):
    rows = json.load(open(f"{O}/sem/eval_student_{P}.json"))["clips"]
    test = [r for r in rows if r["split"] == "test"]
    def agg(k, t):
        v = [r[k] for r in test if r["task"] == t and r.get(k) is not None]
        return np.mean(v) if v else np.nan
    print(f"L{P[1]} B (test, successful rollouts): " + " | ".join(
        f"{t[:5]} succ {100 * agg('success', t):.0f}% n {sum(r['sem_n'] for r in test if r['task'] == t)} "
        f"tempo {1000 * agg('sem_tempo_dev_s', t):.0f}ms miss {100 * agg('sem_td_miss', t):.0f}% "
        f"contact {100 * agg('sem_contact_agree', t):.0f}% stakes {100 * agg('sem_stakes_ok', t):.0f}%" for t in T))
    for F in ("0.5", "1.0"):
        f = f"{O}/frz{F}/eval_student_{P}.json"
        if os.path.exists(f):
            s = json.load(open(f))["summary"]
            print(f"L{P[1]} C freeze {F}s (test): " + "/".join(f"{100 * s[f'{t}/test']['success']:.0f}" for t in T))
EOF
