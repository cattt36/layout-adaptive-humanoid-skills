"""Execution-quality table (CrouchedTraverse, all clips) per level, mean +- std over the given result dirs (seeds)."""
import json
import os
import sys

import numpy as np

KEYS = ["success", "body_err", "peak_force_bw", "max_loading_rate_bw_s", "foot_slip_m_s", "torque_sat_frac",
        "contact_mismatch"]
task = os.environ.get("HNU_TASK", "CrouchedTraverse")
print(f"task {task}: " + " | ".join(KEYS))
for P in ["teacher", "Z0", "Z1", "Z2", "Z3", "Z4"]:
    v = []
    for R in sys.argv[1:]:
        f = os.path.join(R, f"eval_{'teacher_teacher' if P == 'teacher' else 'student_' + P}.json")
        if os.path.exists(f):
            s = json.load(open(f))["summary"].get(f"{task}/all", {})
            if all(k in s for k in KEYS):
                v.append([s[k] for k in KEYS])
    if v:
        v = np.array(v)
        print(f"{P:8s} n={len(v)} " + " | ".join(f"{m:.4g}±{s:.2g}" for m, s in zip(v.mean(0), v.std(0))))
