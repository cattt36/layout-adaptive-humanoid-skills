#!/bin/bash
# Fall-detection sensitivity on TRAIN/VAL trials only: strict (0.40 m / 57 deg) vs loose (0.30 m / 85 deg) instantaneous
# detection; the end-of-clip completion check (pelvis within 1 m of the reference end) is kept in both.
H=/root/gpufree-data/hnu_v2
cd $H/code
PYD=/root/gpufree-data/venvs/deploy/bin/python
for P in Z0 Z4; do
  E=$H/deploy_exports/diag_t_$P
  HNU_FALL_H=0.30 HNU_FALL_TILT=1.48 $PYD deploy/sim2sim.py $E --trials 4 --workers 40 --crit task --out $E/task_loose.json > /dev/null 2>&1
done
$PYD - << 'EOF'
import json, collections
import numpy as np
H = "/root/gpufree-data/hnu_v2"
sp = json.load(open(f"{H}/data/clip_split_trial/split.json"))
where = {c: k for k in ("train", "val", "test") for c in sp[k]}
for P in ("Z0", "Z4"):
    for tag in ("task", "task_loose"):
        rows = [r for r in json.load(open(f"{H}/deploy_exports/diag_t_{P}/{tag}.json"))["rows"] if where.get(r["clip"]) in ("train", "val")]
        cells = []
        for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
            rs = [r for r in rows if r["task"] == t]
            cells.append(f"{t[:6]} {100 * np.mean([r['success'] for r in rs]):.0f}% {dict(collections.Counter(r['cause'] for r in rs if r['cause']))}")
        print(f"L{P[1]} {tag:10s} | " + " | ".join(cells))
EOF