"""DuMuQiao bridge from DuMuQiao.usd + the boundary markers, per clip in robot coordinates (lab-faithful heights).

USD -> lab registration (rotation 180 deg about the vertical, z_lab = -0.98 - z_usd) puts the USD ramp foot, ramp top
and plank end on the rear_lower / rear_upper / front_upper markers (|dz| <= 0.04 m).  Geometry:
  plank  : top surface on the marker line rear_upper -> front_upper, USD width 0.30 m and thickness 0.29 m
  ramp   : rear_lower -> rear_upper board (thickness 0.10 m)
  leg    : front_lower -> front_upper
  posts  : the 5 USD support cylinders (radius from USD) on the plank centre line, floor -> plank underside
           (the last one stands under the ramp: floor -> ramp)
Lab -> robot per clip: similarity fit of the CSV force-plate toe marks to the robot toes (scale s ~ 0.77).
No height adjustment: reports how deep the retargeted reference penetrates each part.

  python make_bridge_usd.py <ref_dir> <scene_csv> <out_dir>
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bridge_check import fit_similarity, read_csv_footholds  # noqa: E402
from make_bridge import FL, FU, RL, RU, box_between, min_dist, quat, to_robot  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import TOE_LOCAL, load_robot  # noqa: E402
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec  # noqa: E402

HALF_W, PLANK_T, BOARD_T = 0.15, 0.29, 0.10
USD_POSTS = [(-2.634, 0.15), (-1.433, 0.10), (-0.234, 0.10), (0.963, 0.15), (2.24, 0.15)]  # (z_usd, radius)
# Capture-video layout (overrides the USD posts / plank cross-section): HNU_POSTS_Z = lab z of the posts,
# HNU_POST_R = post radius, HNU_BEAM = beam (log) diameter.
if os.environ.get("HNU_POSTS_Z"):
    _r = float(os.environ.get("HNU_POST_R", "0.05"))
    USD_POSTS = [(-0.98 - float(z), _r) for z in os.environ["HNU_POSTS_Z"].split(",")]
if os.environ.get("HNU_BEAM"):
    HALF_W = PLANK_T = float(os.environ["HNU_BEAM"])
    HALF_W /= 2


def lab_parts():
    """Parts in lab coordinates: boxes (a, b, half_w, half_t) and cylinders (centre, radius, half_height)."""
    under = lambda z: RU[1] + (z - RU[2]) / (FU[2] - RU[2]) * (FU[1] - RU[1]) - PLANK_T  # noqa: E731
    line_x = lambda z: RU[0] + (z - RU[2]) / (FU[2] - RU[2]) * (FU[0] - RU[0])  # noqa: E731
    parts = {
        "plank": ("box", RU - [0, PLANK_T / 2, 0], FU - [0, PLANK_T / 2, 0], HALF_W, PLANK_T / 2),
        "ramp": ("box", RL, RU, HALF_W, BOARD_T / 2),
        "leg": ("box", FL, FU, HALF_W, BOARD_T / 2),
    }
    for i, (zu, rad) in enumerate(USD_POSTS):
        z = -0.98 - zu
        if RU[2] <= z <= FU[2]:
            top = under(z)
        else:  # under the ramp
            top = RL[1] + (z - RL[2]) / (RU[2] - RL[2]) * (RU[1] - RL[1]) - BOARD_T
        top = max(top, 0.05)
        parts[f"post{i}"] = ("cyl", np.array([line_x(min(max(z, RU[2]), FU[2])), top / 2, z]), rad, top / 2)
    return parts


def robot_parts(s, R, t, parts):
    out = {}
    for k, p in parts.items():
        if p[0] == "box":
            a, b = to_robot(s, R, t, p[1]), to_robot(s, R, t, p[2])
            bx = box_between(a, b, p[3] * s, p[4] * s)
            out[k] = dict(type="box", **bx)
        else:
            c = to_robot(s, R, t, p[1])
            out[k] = dict(type="cyl", center=c.tolist(), radius=p[2] * s, half_height=p[3] * s, yaw=0.0, pitch=0.0)
    return out


def model(parts):
    spec = get_spec()
    for k, b in parts.items():
        g = spec.worldbody.add_geom()
        g.name = k
        if b["type"] == "box":
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, b["half"]
        else:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [b["radius"], b["half_height"], 0]
        g.pos, g.quat = b["center"], quat(b["yaw"], b["pitch"])
        g.contype = g.conaffinity = 0
    m = spec.compile()
    obs = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, k) for k in parts}
    rob = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    return m, mujoco.MjData(m), obs, rob


def main():
    ref_dir, csv_path, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    r = load_robot()
    feet = read_csv_footholds(csv_path)
    lp = lab_parts()
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
        A, B = np.array(A), np.array(B)
        s, R, t = fit_similarity(A, B)
        err = float(np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1).max())
        parts = robot_parts(s, R, t, lp)
        m, d, obs, rob = model(parts)
        dist = min_dist(m, d, obs, rob, Q[::3])
        lift = 0.0
        # raise plank (+ posts under it) until the reference clears it by 3 cm (distance is not linear in the lift
        # when the body passes beside the plank, hence the iteration)
        for _ in range(8 if os.environ.get("HNU_BRIDGE_LIFT", "0") == "1" else 0):
            if dist["plank"] >= 0.025:
                break
            dl = max(0.03 - dist["plank"], 0.02)
            lift += dl
            parts["plank"]["center"][2] += dl
            for k in parts:
                if k.startswith("post") and parts[k]["center"][2] + parts[k]["half_height"] > 0.3:
                    parts[k]["center"][2] += dl / 2
                    parts[k]["half_height"] += dl / 2
            m, d, obs, rob = model(parts)
            dist = min_dist(m, d, obs, rob, Q[::3])
        json.dump(dict(trial=name, scale=float(s), fit_err_m=err, parts=parts, ref_min_dist_m=dist, plank_lift_m=lift),
                  open(os.path.join(out_dir, f"{name}_bridge.json"), "w"), indent=1)
        print(f"{name}: s {s:.3f} fit {err:.3f}  ref min dist " + " ".join(f"{k} {v:+.2f}" for k, v in dist.items()), flush=True)


if __name__ == "__main__":
    main()
