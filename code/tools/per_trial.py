"""Per-trial success (task criterion) of L0 / L4 seed 1: v6g10t vs v6g10j vs v6g10m (missing runs are skipped).
  python per_trial.py [task_prefix] [splits] [dr]   e.g.  per_trial.py FiveStepPoleRun val,test 0"""
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2/runs"
TASK = sys.argv[1] if len(sys.argv) > 1 else ""
SPLITS = (sys.argv[2] if len(sys.argv) > 2 else "test").split(",")
DRS = (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")
RUNS = [(b, P) for b in ("t", "j", "m") for P in ("Z0", "Z4")]
for dr in DRS:
    tab, have = {}, []
    for b, P in RUNS:
        f = f"{H}/results_v6g10{b}_s1_dr{dr}/eval_student_{P}.json"
        if not os.path.exists(f):
            continue
        have.append((b, P))
        for c in json.load(open(f))["clips"]:
            if c["split"] in SPLITS and c["clip"].startswith(TASK):
                tab.setdefault(c["clip"], {})[(b, P)] = c
    print(f"=== dr{dr}: success % (stakes all hit %)   " + " | ".join(f"{b} L{P[1]}" for b, P in have))
    for clip in sorted(tab):
        cells = []
        for r in have:
            c = tab[clip][r]
            h = c.get("stakes_all_hit")
            cells.append(f"{100 * c['success']:4.0f}" + (f" ({100 * h:3.0f})" if h is not None and h == h else "      "))
        print(f"{clip:18s} {next(iter(tab[clip].values()))['split']:5s} " + " | ".join(cells))
