"""Infer where support posts can stand under the DuMuQiao plank from the recorded motion itself.

For a vertical post of radius r on the plank centre line at lab position z, the horizontal clearance of a reference
is min over frames and robot collision geoms of  |xy_geom - xy_post| - (horizontal half-extent of the geom) - r
(robot geoms mapped to lab metres with the per-clip similarity transform).  A post position is "consistent with the
data" if every clip clears it.  Prints the clearance profile along the plank, the feasible intervals, and for each
DuMuQiao.usd post the nearest feasible position.

  python infer_posts.py <ref_dir> <scene_csv> [radius_lab=0.10]
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
rad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
r = load_robot()
m, d = r.m, r.d
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
feet = read_csv_footholds(csv_path)
zs = np.arange(RU[2], FU[2] + 1e-6, 0.05)
line_x = RU[0] + (zs - RU[2]) / (FU[2] - RU[2]) * (FU[0] - RU[0])
P = np.stack([line_x, zs], 1)  # candidate post centres (lab x, z)
clear = []
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
            d.qpos[:] = Q[i]
            mujoco.mj_kinematics(m, d)
            fid = r.foot_ids[0 if foot == "L" else 1]
            A.append([X, Z]); B.append((d.xpos[fid] + d.xmat[fid].reshape(3, 3) @ TOE_LOCAL)[:2])
    A, B = np.array(A), np.array(B)
    s, R, t = fit_similarity(A, B)
    if np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1).max() > 0.1:
        continue
    Rinv = np.linalg.inv(R)
    xy, ext = [], []
    for q in Q[::2]:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        c = d.geom_xpos[geoms][:, :2]
        e = np.einsum("gij,gj->gi", np.abs(d.geom_xmat[geoms].reshape(-1, 3, 3)), m.geom_aabb[geoms, 3:])[:, :2].max(1)
        xy.append((Rinv @ (c - t).T).T / s)
        ext.append(e / s)
    xy, ext = np.concatenate(xy), np.concatenate(ext)
    dist = np.linalg.norm(xy[None] - P[:, None], axis=-1) - ext[None] - rad  # [n_z, n_samples]
    clear.append(dist.min(1))
clear = np.array(clear)  # [clips, n_z]
worst = clear.min(0)
share = (clear > 0).mean(0)
print(f"{len(clear)} clips, post radius {rad} m (lab).  z: worst clearance (m) | share of clips clear")
for z, w, sh in zip(zs[::2], worst[::2], share[::2]):
    print(f"  z {z:+.2f}: {w:+.3f} | {sh:.2f} {'#' * int(round(sh * 20))}")
ok = worst > 0
iv, start = [], None
for z, o in zip(zs, ok):
    if o and start is None:
        start = z
    if not o and start is not None:
        iv.append((start, z - 0.05)); start = None
if start is not None:
    iv.append((start, zs[-1]))
print("feasible post intervals (all clips clear):", [(round(a, 2), round(b, 2)) for a, b in iv])
for zu, rr in USD_POSTS:
    z = -0.98 - zu
    if not RU[2] <= z <= FU[2]:
        continue
    k = int(np.argmin(np.abs(zs - z)))
    feas = zs[ok]
    near = feas[np.argmin(np.abs(feas - z))] if len(feas) else np.nan
    print(f"USD post at z {z:+.2f}: worst clearance {worst[k]:+.3f}, clips clear {share[k]:.2f}, nearest feasible z {near:+.2f}")
