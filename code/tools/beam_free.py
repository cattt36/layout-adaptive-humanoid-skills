"""DEMO (flat CT): where along the beam line can a post stand without being touched?  For the reference beam
line (bridge json plank axis) and a few parallel / rotated variants, the clearance of a post (radius of the json posts) at
every point of the line (min over robot bodies and frames, 2nd percentile over the simulated runs, tools/prop_spread.py
traces) -> free intervals (clearance >= CLR) in the start frame.
  python beam_free.py <export_dir> <motion_npz> <trace_dir> <hold_s> <bridge_json> [CLR]"""
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
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in meta["body_names"]]
ref = np.load(mfile)
T = len(ref["joint_pos"])
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = ref["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T  # noqa: E731
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
XY = []
for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
    qp = np.load(f)["qpos"]
    fr = []
    for k in range(0, T, 2):
        if H + k >= len(qp):
            break
        data.qpos[:] = qp[H + k][:model.nq]
        mujoco.mj_kinematics(model, data)
        fr.append(to_s(data.xpos[bid]))
    XY.append(np.array(fr).reshape(-1, 2))
P = json.load(open(bj))["parts"]
pl = P["plank"]
r = float(P["post0"]["radius"])
c0 = to_s(pl["center"])
L = pl["half"][0]
print(f"beam centre ({c0[0]:.2f}, {c0[1]:.2f}), length {2 * L:.2f} m, post radius {r:.3f}, clearance >= {CLR} m, {len(XY)} runs")
for dyaw in (0, -10, 10):
    for off in (0.0, -0.2, 0.2, -0.4, 0.4):
        yw = pl["yaw"] - a0 + np.radians(dyaw)
        ax = np.array([np.cos(yw), np.sin(yw)])
        nrm = np.array([-ax[1], ax[0]])
        ts = np.arange(-L - 0.3, L + 0.301, 0.05)
        cl = []
        for t in ts:
            c = c0 + off * nrm + t * ax
            cl.append(np.percentile([np.min(np.linalg.norm(x - c, axis=-1)) - r for x in XY], 2))
        free = np.array(cl) >= CLR
        iv, s = [], None
        for i, fr in enumerate(free):
            if fr and s is None:
                s = i
            if (not fr or i == len(free) - 1) and s is not None:
                e = i if fr else i - 1
                iv.append((ts[s], ts[e]))
                s = None
        pts = lambda t: c0 + off * nrm + t * ax  # noqa: E731
        desc = ", ".join(f"[{a:+.2f}, {b:+.2f}] ({pts(a)[0]:.2f},{pts(a)[1]:.2f})->({pts(b)[0]:.2f},{pts(b)[1]:.2f})" for a, b in iv)
        print(f"rot {dyaw:+3d} deg  shift {off:+.1f} m: free along the line (t from the centre, start-frame xy): {desc or 'none'}")
