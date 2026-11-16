"""CrouchedTraverse tracking error (mean body position error, m) and success, per level, averaged over the given result dirs."""
import json
import os
import sys

import numpy as np

rows = {}
for R in sys.argv[1:]:
    for P in ["teacher", "Z0", "Z1", "Z2", "Z3", "Z4"]:
        f = os.path.join(R, f"eval_{'teacher_teacher' if P == 'teacher' else 'student_' + P}.json")
        if os.path.exists(f):
            s = json.load(open(f))["summary"]["CrouchedTraverse/all"]
            sp = json.load(open(f.replace(".json", "_push.json")))["summary"]["CrouchedTraverse/all"]
            rows.setdefault(P, []).append((s["success"], s["body_err"], sp["success"], sp["body_err"]))
for P, v in rows.items():
    v = np.array(v)
    m, sd = v.mean(0), v.std(0)
    print(f"{P:8s} n={len(v)}  success {100*m[0]:.1f}±{100*sd[0]:.1f}  body_err {100*m[1]:.1f} cm  "
          f"| push success {100*m[2]:.1f}±{100*sd[2]:.1f}  body_err {100*m[3]:.1f} cm")
