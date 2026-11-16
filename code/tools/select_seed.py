"""Deployment model selection (pre-registered rule, identical for every level):
score = mean over the three actions of the VALIDATION-split success in the randomised environment, clean and push
weighted equally; if the best two seeds are within 2 points, prefer the lower actuator saturation.
The test split is never used here.

  python select_seed.py <results_prefix>   (dirs <prefix>_s{1,2,3}_dr1 with eval_student_Zk[_push].json)
"""
import json
import os
import sys

import numpy as np

pre = sys.argv[1]
TASKS = ["ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"]
out = {}
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    cands = []
    for s in (1, 2, 3, 4, 5):  # missing seeds are skipped
        d = f"{pre}_s{s}_dr1"
        fc, fp = os.path.join(d, f"eval_student_{P}.json"), os.path.join(d, f"eval_student_{P}_push.json")
        if not (os.path.exists(fc) and os.path.exists(fp)):
            continue
        sc, sp = json.load(open(fc))["summary"], json.load(open(fp))["summary"]
        val = [0.5 * (sc[f"{t}/val"]["success"] + sp[f"{t}/val"]["success"]) for t in TASKS if f"{t}/val" in sc]
        sat = np.mean([sc[f"{t}/all"].get("torque_sat_frac", 0.0) for t in TASKS if f"{t}/all" in sc])
        cands.append(dict(seed=s, score=float(np.mean(val)), torque_sat=float(sat)))
    if not cands:
        continue
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]
    if len(cands) > 1 and cands[0]["score"] - cands[1]["score"] < 0.02 and cands[1]["torque_sat"] < cands[0]["torque_sat"]:
        best = cands[1]
    out[P] = dict(selected_seed=best["seed"], candidates=cands)
    print(f"L{P[1]}: seed {best['seed']}  " + "  ".join(f"s{c['seed']}: val {100 * c['score']:.1f}% sat {c['torque_sat']:.4f}" for c in cands))
json.dump(out, open(f"{pre}_selection.json", "w"), indent=1)
