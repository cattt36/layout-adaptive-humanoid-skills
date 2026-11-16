"""DEMO (flat CT): a bridge the robot does not touch.  The slalom path crosses the reference beam line almost
everywhere, so the props become ONE beam on TWO end posts.  Candidate beam lines = the reference plank axis rotated
(-20..+20 deg) and shifted (-0.6..+0.6 m); on each line the two posts go to the widest pair of free positions (post clearance
>= CLR, 2nd percentile over the simulated runs) the beam spans (>= 3.0 m apart, <= beam length - 0.1 m).  For every feasible line:
  required beam underside = max over runs (98th pct) of the robot top under the beam (geom centre z + bounding radius
  of geoms within the beam half width + rbound horizontally) + MARGIN; also how long the robot is under the beam
  (crouched part of the clip) -- the line should be crossed while crouched, i.e. the required height stays low.
Prints the best lines (lowest required underside first).
  python beam_plan.py <export_dir> <motion_npz> <trace_dir> <hold_s> <bridge_json> [CLR] [MARGIN]"""
import glob
import json
import os
import sys

import mujoco
import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


ex, mfile, tdir, hold_s, bj = sys.argv[1:6]
CLR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
MARGIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.08
MINSPAN = 3.0  # m between the two posts (the reference beam is ~3.9 m)
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in meta["body_names"]]
rob_bodies = set()
for b in range(model.nbody):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if nm.startswith("robot/"):
        rob_bodies.add(b)
gids = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in rob_bodies])
grb = model.geom_rbound[gids]
ref = np.load(mfile)
T = len(ref["joint_pos"])
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = ref["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T  # noqa: E731
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
BXY, GXY, GZ = [], [], []  # per run: body xy [F, B, 2]; geom xy [F, G, 2]; geom top z [F, G]
for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
    qp = np.load(f)["qpos"]
    b, g, z = [], [], []
    for k in range(0, T, 2):
        if H + k >= len(qp):
            break
        data.qpos[:] = qp[H + k][:model.nq]
        mujoco.mj_kinematics(model, data)
        b.append(to_s(data.xpos[bid]))
        g.append(to_s(data.geom_xpos[gids]))
        z.append(data.geom_xpos[gids, 2] + grb)
    BXY.append(np.array(b)), GXY.append(np.array(g)), GZ.append(np.array(z))
P = json.load(open(bj))["parts"]
pl = P["plank"]
r = float(P["post0"]["radius"])
bw = float(pl["half"][1])
c0 = to_s(pl["center"])
L = float(pl["half"][0])
ref_under = pl["center"][2] - pl["half"][2]


def post_clr(c):
    return float(np.percentile([np.min(np.linalg.norm(x - c, axis=-1)) - r for x in BXY], 2))


res = []
for dyaw in range(-20, 21, 5):
    for off in np.arange(-0.6, 0.61, 0.1):
        yw = pl["yaw"] - a0 + np.radians(dyaw)
        ax = np.array([np.cos(yw), np.sin(yw)])
        nrm = np.array([-ax[1], ax[0]])
        ts = np.arange(-L - 0.6, L + 0.601, 0.05)
        free = [t for t in ts if post_clr(c0 + off * nrm + t * ax) >= CLR]
        lo = [t for t in free if t < 0]
        hi = [t for t in free if t > 0]
        if not lo or not hi:
            continue
        # widest post pair that the beam still spans (beam length - 0.1 m): the beam must cover the crouched passage
        pairs = [(a, b) for a in lo for b in hi if b - a <= 2 * L - 0.10]
        if not pairs:
            continue
        t1, t2 = max(pairs, key=lambda p: p[1] - p[0])
        if t2 - t1 < MINSPAN:
            continue
        p1, p2 = c0 + off * nrm + t1 * ax, c0 + off * nrm + t2 * ax
        # robot top under the beam segment [t1, t2]
        tops = []
        for g, z in zip(GXY, GZ):
            d = g - p1
            s = d @ ax
            lat = np.abs(d @ nrm)
            under = (s > 0) & (s < t2 - t1) & (lat < bw + grb[None])
            tops.append(float(np.max(np.where(under, z, 0.0))))
        need = float(np.percentile(tops, 98)) + MARGIN
        res.append((need, dyaw, off, p1, p2, t2 - t1))
res.sort(key=lambda x: x[0])
print(f"reference beam: centre ({c0[0]:.2f}, {c0[1]:.2f}), length {2 * L:.2f} m, underside {ref_under:.2f} m; "
      f"post clearance >= {CLR} m (body origins), height margin {MARGIN} m; {len(BXY)} runs")
print("| beam underside needed (m) | rot (deg) | shift (m) | post A x, y | post B x, y | span (m) |\n|---|---|---|---|---|---|")
for need, dyaw, off, p1, p2, sp in res[:12]:
    print(f"| {need:.2f} | {dyaw:+d} | {off:+.1f} | {p1[0]:.2f}, {p1[1]:.2f} | {p2[0]:.2f}, {p2[1]:.2f} | {sp:.2f} |")
if not res:
    print("no feasible two-post line")
