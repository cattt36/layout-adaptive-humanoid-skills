#!/bin/bash
# Held-out students (v6g10h L0 no aug; pilot L0+aug, L4+aug, explicit reference) under the TRACKING criterion
# (training-evaluation terminations) and the TASK criterion (no fall, stakes hit, obstacle passed), sim2sim nominal.
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 PYTHONPATH=$H/code
PYM=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
declare -A ST=([L0]="$H/runs/students_v6g10h_d2_s1/Z0/student_2000.pt Z0" [L0A]="$H/runs/students_v6g10p_d2_s1/Z0A/student_1000.pt Z0" \
               [L4A]="$H/runs/students_v6g10p_d2_s1/Z4A/student_1000.pt Z4" [REF]="$H/runs/students_v6g10p_d2_s1/REF/student_1000.pt Z0")
for k in L0 L0A L4A REF; do
  set -- ${ST[$k]}
  E=$H/deploy_exports/diag_heldout_$k; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=0 $PYM deploy/export_student.py --student $1 --latent-dir $H/data/z_tables_v5/$2 --out $E > $E/export.log 2>&1
  for C in track task; do
    [ -f $E/$C.json ] || $PYD deploy/sim2sim.py $E --trials 2 --workers 40 --crit $C --out $E/$C.json > /dev/null 2>&1
  done
done
$PYD - << 'EOF'
import json, collections, os
H = "/root/gpufree-data/hnu_v2"
sp = {}
for s in ("train", "val", "test"):
    for c in open(f"{H}/data/clip_split_v5/clips_{s}.txt").read().split():
        sp[c] = s
for C in ("track", "task"):
    print(f"\n== criterion: {C}  (success % train / val / test; test failure causes)")
    for k in ("L0", "L0A", "L4A", "REF"):
        f = f"{H}/deploy_exports/diag_heldout_{k}/{C}.json"
        if not os.path.exists(f):
            continue
        rows = json.load(open(f))["rows"]
        cells = []
        for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
            r = {s: [x for x in rows if x["task"] == t and sp.get(x["clip"]) == s] for s in ("train", "val", "test")}
            cells.append(f"{t[:6]} " + "/".join(f"{100 * sum(x['success'] for x in v) / max(len(v), 1):.0f}" for v in r.values()))
        causes = collections.Counter(x["cause"] for x in rows if sp.get(x["clip"]) == "test" and x["cause"])
        print(f"  {k:4s} " + " | ".join(cells) + f" | test causes {dict(causes)}")
EOF
