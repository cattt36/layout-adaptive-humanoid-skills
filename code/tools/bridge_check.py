"""Exact clearance of CrouchedTraverse motions (reference and expert rollouts) w.r.t. the registered, scaled bridge.

Registration (lab frame, metres, y up): DuMuQiao.usd rotated 180 deg about the vertical axis, z_lab = Z_OFF - z_usd.
Evidence: rigid body "Bridge_Da" in the mocap CSV has identity orientation at x ~ 0; the flat plank end
(z_usd = -3.25) lands on the "front-up" mark (z = 2.23) and the ramp foot (z_usd = 3.13) on "terminal-down" (z = -4.07).
Lab -> robot per trial: similarity transform fitted between CSV toe positions on the six plates and the robot toe
positions at the same frames (fit error reported).  Obstacles are added as MuJoCo geoms (box plank, sloped box ramp,
cylinder posts) and mj_geomDistance gives exact signed distances to every robot collision geom.
"""
import csv, glob, json, os, sys
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import TOE_LOCAL, load_robot  # noqa: E402
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec  # noqa: E402

Z_OFF = -0.98
def zl(zu):
    return Z_OFF - zu
PLANK = dict(z0=zl(1.63), z1=zl(-3.25), y0=1.011, y1=1.301, half_w=0.15)       # full-height plank (lab)
RAMP = dict(za=zl(1.63), ya=1.156, zb=zl(3.13), yb=0.0, thick=0.29, half_w=0.15)  # sloped end down to the floor
POSTS = [(zl(-2.634), 0.15, 1.151), (zl(-1.433), 0.10, 1.151), (zl(-0.234), 0.10, 1.151), (zl(0.963), 0.15, 1.151),
         (zl(2.24), 0.15, 0.818)]


def read_csv_footholds(path):
    rows = {}
    with open(path, encoding="gbk") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows[row[0]] = [("L" if "LToe" in row[1 + 6 * i] else "R", int(float(row[3 + 6 * i])),
                             float(row[4 + 6 * i]) / 1000, float(row[6 + 6 * i]) / 1000) for i in range(6)]
    return rows


def fit_similarity(A, B):
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    R = U @ Vt
    s = S.sum() / (a ** 2).sum()
    return s, R, mb - s * R @ ma


def lab_to_robot(s, R, t, x, y, z):
    xy = s * (R @ np.array([x, z])) + t
    return np.array([xy[0], xy[1], s * y])


def build_model(s, R, t):
    spec = get_spec()
    wb = spec.worldbody
    ax = R @ np.array([0.0, 1.0])  # lab +Z direction in robot xy
    yaw = np.arctan2(ax[1], ax[0])
    names = []

    def add_box(name, center, half, yaw, pitch=0.0):
        g = wb.add_geom()
        g.name, g.type = name, mujoco.mjtGeom.mjGEOM_BOX
        g.size = half
        g.pos = center
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([0.0, pitch, yaw]), "xyz")
        g.quat = q
        g.contype = g.conaffinity = 0
        names.append(name)

    zc = 0.5 * (PLANK["z0"] + PLANK["z1"])
    add_box("plank", lab_to_robot(s, R, t, 0, 0.5 * (PLANK["y0"] + PLANK["y1"]), zc),
            [0.5 * abs(PLANK["z1"] - PLANK["z0"]) * s, PLANK["half_w"] * s, 0.5 * (PLANK["y1"] - PLANK["y0"]) * s], yaw)
    L = np.hypot(RAMP["zb"] - RAMP["za"], RAMP["yb"] - RAMP["ya"])
    pitch = np.arctan2(RAMP["ya"] - RAMP["yb"], abs(RAMP["zb"] - RAMP["za"]))
    rc = lab_to_robot(s, R, t, 0, 0.5 * (RAMP["ya"] + RAMP["yb"]), 0.5 * (RAMP["za"] + RAMP["zb"]))
    sign = np.sign(RAMP["zb"] - RAMP["za"])
    add_box("ramp", rc, [0.5 * L * s, RAMP["half_w"] * s, 0.5 * RAMP["thick"] * s], yaw, pitch * sign)
    for i, (z0, r, top) in enumerate(POSTS):
        g = wb.add_geom()
        g.name, g.type = f"post{i}", mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r * s, 0.5 * top * s, 0]
        g.pos = lab_to_robot(s, R, t, 0, 0.5 * top, z0)
        g.contype = g.conaffinity = 0
        names.append(f"post{i}")
    m = spec.compile()
    obs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
    rob = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    return m, mujoco.MjData(m), obs, rob, names


def clearance(m, d, obs, rob, Q):
    """per frame: min signed distance to [plank+ramp], [posts]"""
    ft = np.zeros(6)
    out = np.zeros((len(Q), 2))
    for i, q in enumerate(Q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        best = [1.0, 1.0]
        for k, o in enumerate(obs):
            grp = 0 if k < 2 else 1
            for g in rob:
                dist = mujoco.mj_geomDistance(m, d, g, o, 0.5, ft)
                if dist < best[grp]:
                    best[grp] = dist
        out[i] = best
    return out


def main():
    ref_dir, csv_path, roll_dir, out = sys.argv[1:5]
    r = load_robot()
    feet = read_csv_footholds(csv_path)
    res = []
    for f in sorted(glob.glob(os.path.join(ref_dir, "CrouchedTraverse_*_Dynamics.npz"))):
        name = os.path.basename(f).replace("_Dynamics.npz", "")
        if name not in feet:
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
        fit_err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        m, d, obs, rob, _ = build_model(s, R, t)
        c = clearance(m, d, obs, rob, Q[::3])
        rec = dict(trial=name, scale=round(float(s), 3), fit_err_max_m=round(float(fit_err.max()), 3),
                   ref_plank_min_m=round(float(c[:, 0].min()), 3), ref_post_min_m=round(float(c[:, 1].min()), 3),
                   ref_frames_in_collision_pct=round(float((c.min(1) < 0).mean() * 100), 1))
        rp = os.path.join(roll_dir, f"rollout_{name}.npz")
        if os.path.exists(rp):
            ro = np.load(rp)
            mins, hits = [], 0
            for e in range(ro["root_pos"].shape[1]):
                v = ro["valid"][:, e]
                off = ro["root_pos"][0, e] - Q[0, :3]
                qq = np.concatenate([ro["root_pos"][v, e] - off, ro["root_quat"][v, e], ro["joint_pos"][v, e]], 1)[::3]
                cc = clearance(m, d, obs, rob, qq)
                mins.append(cc.min(0)); hits += int(cc.min() < 0)
            mins = np.array(mins)
            rec.update(roll_plank_min_m=round(float(mins[:, 0].min()), 3), roll_post_min_m=round(float(mins[:, 1].min()), 3),
                       roll_collision_rate=round(hits / len(mins), 3))
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
