"""Run-up statistics per clip (robot-scale stake references): time from clip start to the first stake stance, pelvis
speed at the end of the run-up, and where the run-up starts (time when the pelvis first exceeds 0.3 m/s)."""
import glob
import json
import os
import sys

import numpy as np

d = sys.argv[1]
for task in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    rows = []
    for j in sorted(glob.glob(os.path.join(d, task, "*_stakes.json"))):
        meta = json.load(open(j))
        m = np.load(j.replace("_stakes.json", "_motion.npz"))
        p = m["body_pos_w"][:, 0, :2]
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) * 50.0
        t1 = meta["stakes"][0]["stance"][0] / meta["fps"]
        f1 = int(t1 * 50)
        move = int(np.argmax(v > 0.3))
        rows.append((t1, move / 50.0, v[max(f1 - 10, 0):f1].mean() if f1 > 0 else 0, len(p) / 50.0,
                     np.linalg.norm(p[f1] - p[move]) if f1 > move else 0))
    r = np.array(rows)
    print(f"{task}: n={len(r)} first stake at {np.median(r[:, 0]):.2f}s (median), run-up starts {np.median(r[:, 1]):.2f}s, "
          f"run-up duration {np.median(r[:, 0] - r[:, 1]):.2f}s, run-up distance {np.median(r[:, 4]):.2f} m, "
          f"speed before first stake {np.median(r[:, 2]):.2f} m/s, clip length {np.median(r[:, 3]):.2f}s")
