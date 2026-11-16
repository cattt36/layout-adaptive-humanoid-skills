"""Where along the bridge (lab z) does the retargeted reference's pelvis cross the plank centre line, and how far
does it stay from the line elsewhere?  Compared with the USD post positions (lab z) under the 180-deg registration.

  python crossing_stats.py <ref_dir> <scene_csv>
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bridge_check import fit_similarity, read_csv_footholds  # noqa: E402
from make_bridge import FU, RU  # noqa: E402
from make_bridge_usd import USD_POSTS  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import TOE_LOCAL, load_robot  # noqa: E402

ref_dir, csv_path = sys.argv[1:3]
r = load_robot()
pel = mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
feet = read_csv_footholds(csv_path)
cross, offs = [], []
for name in sorted(feet):
    f = os.path.join(ref_dir, f"{name}_Dynamics.npz")
    if not os.path.exists(f):
        continue
    dd = np.load(f)
    Q, crop0 = dd["qpos"], int(dd["source_crop"][0])
    A, B = [], []
    for foot, frame, X, Z in feet[name]:
        i = frame - crop0
        if 0 <= i < len(Q):
            r.d.qpos[:] = Q[i]
            mujoco.mj_kinematics(r.m, r.d)
            fid = r.foot_ids[0 if foot == "L" else 1]
            A.append([X, Z]); B.append((r.d.xpos[fid] + r.d.xmat[fid].reshape(3, 3) @ TOE_LOCAL)[:2])
    s, R, t = fit_similarity(np.array(A), np.array(B))
    if np.linalg.norm((s * (R @ np.array(A).T)).T + t - np.array(B), axis=1).max() > 0.1:
        continue
    lab = []
    for q in Q:
        r.d.qpos[:] = q
        mujoco.mj_kinematics(r.m, r.d)
        lab.append(np.linalg.inv(R) @ (r.d.xpos[pel][:2] - t) / s)
    lab = np.array(lab)
    u = (lab[:, 1] - RU[2]) / (FU[2] - RU[2])
    dx = lab[:, 0] - (RU[0] + u * (FU[0] - RU[0]))
    inside = (u > 0) & (u < 1)
    sgn = np.sign(dx)
    idx = np.where(inside[1:] & (sgn[1:] != sgn[:-1]))[0]
    cz = lab[idx, 1]
    cross += cz.tolist()
    offs.append(np.median(np.abs(dx[inside])))
    print(f"{name}: pelvis crosses the plank line at lab z = {np.round(cz, 2).tolist()}  median |dx| under bridge {offs[-1]:.2f} m")
posts = sorted(-0.98 - z for z, _ in USD_POSTS)
print("USD posts (lab z):", np.round(posts, 2).tolist())
h, e = np.histogram(cross, bins=np.arange(-4.2, 2.5, 0.3))
print("crossing histogram (lab z bins of 0.3 m):")
for a, n in zip(e[:-1], h):
    print(f"  {a:+.1f}..{a + 0.3:+.1f}: {'#' * n}")
