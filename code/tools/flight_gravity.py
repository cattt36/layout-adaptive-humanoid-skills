"""Is the retargeted reference ballistically consistent?  During flight phases (no foot on the ground) the centre of
mass of a real body accelerates at -g.  Fit the pelvis vertical acceleration over each flight (quadratic fit) and
report the implied gravity per task.  Geometric scaling (s ~ 0.77) with unchanged timing implies g_eff ~ s * g.

  python flight_gravity.py <ref_root> [tasks...]
"""
import glob
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import load_robot  # noqa: E402

root = sys.argv[1]
tasks = sys.argv[2:] or ["ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"]
r = load_robot()
for task in tasks:
    gs, durs = [], []
    for f in sorted(glob.glob(os.path.join(root, task, "*_Dynamics.npz")))[:20]:
        d = np.load(f)
        Q, fps = d["qpos"], float(d["fps"])
        T = len(Q)
        sole = np.zeros((T, 2))
        com = np.zeros(T)
        for t in range(T):
            r.d.qpos[:] = Q[t]
            mujoco.mj_kinematics(r.m, r.d)
            mujoco.mj_comPos(r.m, r.d)
            sole[t] = [r.d.xpos[fid][2] for fid in r.foot_ids]
            com[t] = r.d.subtree_com[0][2]
        air = (sole > 0.08).all(1)
        t = 0
        while t < T:
            if air[t]:
                a = t
                while t < T and air[t]:
                    t += 1
                n = t - a
                if n >= int(0.12 * fps):
                    tt = np.arange(n) / fps
                    c = np.polyfit(tt, com[a:t], 2)
                    gs.append(-2 * c[0]); durs.append(n / fps)
            t += 1
    gs = np.array(gs)
    print(f"{task}: {len(gs)} flights >= 0.12 s, duration median {np.median(durs) if durs else 0:.2f} s, "
          f"implied gravity median {np.median(gs) if len(gs) else float('nan'):.2f} m/s^2 "
          f"(IQR {np.percentile(gs, 25) if len(gs) else 0:.2f}-{np.percentile(gs, 75) if len(gs) else 0:.2f}; real 9.81)")
