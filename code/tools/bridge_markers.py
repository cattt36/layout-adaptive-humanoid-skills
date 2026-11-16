"""Marker-defined DuMuQiao bridge vs the retargeted robot references.

Geometry (lab, Y up, Z forward; LLM_annotation environment.boundary_markers_m): plank top along
rear_upper -> front_upper (width 0.30 m from DuMuQiao.usd), supports rear_lower -> rear_upper (ramp) and
front_lower -> front_upper (front leg).  Lab -> robot per trial: similarity fit between the CSV force-plate toe marks
and the robot toe positions at the same frames (bridge_check.py).

For every trial: the highest point of the robot's collision geoms (conservative: centre + bounding radius) while it
is horizontally inside the plank footprint, in lab metres (divided by the trial scale).  The plank underside must be
above it for the reference to be collision-free; the plank top from the markers is reported for comparison.
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bridge_check import fit_similarity, read_csv_footholds  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import TOE_LOCAL, load_robot  # noqa: E402

RL, RU = np.array([-0.209, 0.085, -4.072]), np.array([-0.157, 1.129, -2.617])
FU, FL = np.array([0.017, 1.256, 2.227]), np.array([0.101, 0.051, 2.061])
HALF_W = 0.15


def main():
    ref_dir, csv_path, out = sys.argv[1:4]
    r = load_robot()
    m, d = r.m, r.d
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
             and not any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") for k in ("ankle",))]
    feet = read_csv_footholds(csv_path)
    res = []
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
        err = float(np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1).max())
        Rinv = np.linalg.inv(R)
        need, n_in, u_at = -1.0, 0, None
        prof = np.full(10, -1.0)  # needed underside per tenth of the plank length
        for q in Q[::2]:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            for g in geoms:
                p = d.geom_xpos[g]
                Rg = d.geom_xmat[g].reshape(3, 3)
                half = m.geom_aabb[g, 3:]
                ext = np.abs(Rg) @ half  # world-axis half extents of the geom's oriented box
                lab = (Rinv @ (p[:2] - t)) / s  # (x, z) in lab
                u = (lab[1] - RU[2]) / (FU[2] - RU[2])
                if not 0 <= u <= 1:
                    continue
                lx = RU[0] + u * (FU[0] - RU[0])
                if abs(lab[0] - lx) < HALF_W + max(ext[0], ext[1]) / s:
                    n_in += 1
                    top = (p[2] + ext[2]) / s
                    k = min(int(u * 10), 9)
                    prof[k] = max(prof[k], top)
                    if top > need:
                        need, u_at = top, u
        plank_top = None if u_at is None else RU[1] + u_at * (FU[1] - RU[1])
        rec = dict(trial=name, scale=round(float(s), 3), fit_err_m=round(err, 3), geom_samples_under_plank=n_in,
                   needed_underside_lab_m=round(float(need), 3), plank_top_there_lab_m=None if plank_top is None else round(float(plank_top), 3),
                   u_at=None if u_at is None else round(float(u_at), 2), profile_by_tenth=np.round(prof, 2).tolist())
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
