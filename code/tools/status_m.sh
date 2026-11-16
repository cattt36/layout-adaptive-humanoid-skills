#!/bin/bash
# Supervision snapshot of the v6g10m batch: batch log, running jobs + iteration, errors, GPUs, finished results.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=${1:-v6g10m}
echo "=== $(date '+%m-%d %H:%M') batch log"; tail -8 $N/$V.log
echo "=== running"
for f in $(ls -t $N/${V}_s*_E_*.log $N/${V}_d2_s*_*.log 2>/dev/null | head -4); do
  echo "$(basename $f): $(grep '^{' $f | tail -1 | cut -c1-110)  errors=$(grep -ciE 'traceback|out of memory|nan' $f)"
done
pgrep -f "run_$V.sh" > /dev/null && echo "pipeline alive" || echo "PIPELINE NOT RUNNING"
pgrep -f "monitor_tb.py $V" > /dev/null && echo "monitor alive" || echo "MONITOR NOT RUNNING"
pgrep -f "tensorboard --logdir" > /dev/null && echo "tensorboard alive" || echo "TENSORBOARD NOT RUNNING"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /root/gpufree-data | tail -1
echo "=== failures in batch log"; grep -i failed $N/$V.log || echo none
cd $H/code
/root/gpufree-data/venvs/mjlab/bin/python - "$V" << 'EOF'
import json, os, sys
import numpy as np
V = sys.argv[1]
H = "/root/gpufree-data/hnu_v2/runs"
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
print("=== test success % DuMu/W1/W2: nominal | randomised | nom+push  [val nominal]  W1/W2 all-hit %, centre cm")
for P in ["Z0", "Z1", "Z2", "Z3", "Z4", "NL", "REF"]:
    for s in range(1, 6):
        r = {}
        for c, dr, x in (("n", 0, ""), ("r", 1, ""), ("p", 0, "_push")):
            f = f"{H}/results_{V}_s{s}_dr{dr}/eval_student_{P}{x}.json"
            if os.path.exists(f):
                r[c] = json.load(open(f))["summary"]
        if "n" not in r:
            continue
        g = lambda c, sp: "/".join(f"{100 * r[c][f'{t}/{sp}']['success']:.0f}" for t in T) if c in r else "-"
        n = r["n"]
        hit = "/".join(f"{100 * (n[f'{t}/test'].get('stakes_all_hit') or 0):.0f}" for t in T[1:])
        dist = "/".join(f"{100 * (n[f'{t}/test'].get('stake_centre_dist_m') or 0):.1f}" for t in T[1:])
        print(f"{P:3s} s{s}: {g('n', 'test'):10s} | {g('r', 'test'):10s} | {g('p', 'test'):10s} [val {g('n', 'val')}]  hit {hit} dist {dist}")
EOF
