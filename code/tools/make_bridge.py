"""Per-clip DuMuQiao bridge in robot coordinates, from the boundary markers (LLM_annotation environment).

Lab geometry (Y up, Z forward): plank top rear_upper -> front_upper (width 0.30 m, DuMuQiao.usd); rear support =
sloped board rear_lower -> rear_upper; front support = leg front_lower -> front_upper.  No intermediate posts (not
in the marker set, and the actor walks along the plank's centre line under it).
Lab -> robot per clip: similarity fit (scale s, yaw R, offset t) of the CSV force-plate toe marks to the robot toes.

Heights: the marker plank top scaled to the robot is s * y_top.  The retargeted reference does not crouch as deep
as the actor (relative to body size), so the plank underside is placed at the reference's own clearance
(highest collision-geom point inside the plank footprint + MARGIN), reported together with the scaled marker height.
Output <clip>_bridge.json: boxes (centre, half sizes, yaw, pitch) in motion coordinates, fit error, clearances.

  python make_bridge.py <ref_dir> <scene_csv> <out_dir>
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
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec  # noqa: E402

RL, RU = np.array([-0.209, 0.085, -4.072]), np.array([-0.157, 1.129, -2.617])
FU, FL = np.array([0.017, 1.256, 2.227]), np.array([0.101, 0.051, 2.061])
HALF_W, THICK, MARGIN = 0.15, 0.10, 0.03
SUPPORT_THICK = 0.10  # lab metres, ramp board / front leg


def to_robot(s, R, t, p):  # lab (x, y, z) -> robot (x, y, z)
    xy = s * (R @ np.array([p[0], p[2]])) + t
    return np.array([xy[0], xy[1], s * p[1]])


def box_between(a, b, half_w, half_t):
    """Box from point a to point b (robot frame) with given half width / half thickness; yaw/pitch of the a->b axis."""
    c = 0.5 * (a + b)
    v = b - a
    L = np.linalg.norm(v)
    yaw = np.arctan2(v[1], v[0])
    pitch = -np.arctan2(v[2], np.hypot(v[0], v[1]))
    return dict(center=c.tolist(), half=[0.5 * L, half_w, half_t], yaw=float(yaw), pitch=float(pitch))


def quat(yaw, pitch):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([yaw, pitch, 0.0]), "zyx")
    return q


def clearance_model(boxes):
    spec = get_spec()
    for k, b in boxes.items():
        g = spec.worldbody.add_geom()
        g.name, g.type = k, mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.quat = b["half"], b["center"], quat(b["yaw"], b["pitch"])
        g.contype = g.conaffinity = 0
    m = spec.compile()
    obs = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, k) for k in boxes}
    rob = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    return m, mujoco.MjData(m), obs, rob


def min_dist(m, d, obs, rob, Q):
    ft = np.zeros(6)
    out = {k: 9.0 for k in obs}
    for q in Q:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for k, o in obs.items():
            for g in rob:
                out[k] = min(out[k], mujoco.mj_geomDistance(m, d, g, o, 1.0, ft))
    return out


def main():
    ref_dir, csv_path, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    r = load_robot()
    feet = read_csv_footholds(csv_path)
    rows = []
    for name in sorted(feet):
        f = os.path.join(ref_dir, f"{name}_Dynamics.npz")
        if not os.path.exists(f):
            continue
        dd = np.load(f)
        Q, crop0, fps = dd["qpos"], int(dd["source_crop"][0]), float(dd["fps"])
        A, B = [], []
        for foot, frame, X, Z in feet[name]:
            i = frame - crop0
            if 0 <= i < len(Q):
                r.d.qpos[:] = Q[i]
                mujoco.mj_kinematics(r.m, r.d)
                fid = r.foot_ids[0 if foot == "L" else 1]
                A.append([X, Z]); B.append((r.d.xpos[fid] + r.d.xmat[fid].reshape(3, 3) @ TOE_LOCAL)[:2])
        A, B = np.array(A), np.array(B)
        s, R, t = fit_similarity(A, B)
        err = float(np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1).max())
        ru, fu, rl, fl = (to_robot(s, R, t, p) for p in (RU, FU, RL, FL))
        # supports first (fixed by the markers), plank placed high, then lowered to the reference clearance
        sup = dict(ramp=box_between(rl, ru, HALF_W * s, 0.5 * SUPPORT_THICK * s),
                   leg=box_between(fl, fu, HALF_W * s, 0.5 * SUPPORT_THICK * s))
        probe = dict(sup, plank=box_between(ru + [0, 0, 5.0], fu + [0, 0, 5.0], HALF_W * s, 0.01))
        m, d, obs, rob = clearance_model(probe)
        Qs = Q[::3]
        # highest robot point under the plank footprint: lower a thin probe plank until it touches the reference
        lo, hi = 0.3, 2.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            a = np.array([ru[0], ru[1], mid])
            b = np.array([fu[0], fu[1], mid + fu[2] - ru[2]])  # same slope as the marker plank
            pb = dict(probe, plank=box_between(a, b, HALF_W * s, 0.005))
            m, d, obs, rob = clearance_model(pb)
            dist = min_dist(m, d, {"plank": obs["plank"]}, rob, Qs)["plank"]
            lo, hi = (mid, hi) if dist < 0 else (lo, mid)
        under_rear = hi + MARGIN  # underside height at the rear end of the plank (robot m); slope kept from markers
        dz = under_rear - ru[2]
        plank = box_between(ru + [0, 0, dz + THICK * s / 2], fu + [0, 0, dz + THICK * s / 2], HALF_W * s, THICK * s / 2)
        boxes = dict(sup, plank=plank)
        m, d, obs, rob = clearance_model(boxes)
        dist = min_dist(m, d, obs, rob, Qs)
        meta = dict(trial=name, scale=float(s), fit_err_m=err, boxes=boxes,
                    plank_underside_rear_robot_m=float(under_rear), plank_top_rear_marker_scaled_m=float(ru[2]),
                    lift_vs_markers_robot_m=float(dz + THICK * s), ref_min_dist_m={k: float(v) for k, v in dist.items()})
        json.dump(meta, open(os.path.join(out_dir, f"{name}_bridge.json"), "w"), indent=1)
        rows.append(meta)
        print(f"{name}: s {s:.3f} fit {err:.3f}  underside(rear) {under_rear:.2f} m vs marker top scaled {ru[2]:.2f} m "
              f"(plank lifted {dz + THICK * s:+.2f} m)  ref min dist plank {dist['plank']:.3f} ramp {dist['ramp']:.3f} leg {dist['leg']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
