"""DEMO: how the CT reference weaves around the 4 posts (closest-approach order + side), and the same for the
robot's simulated pelvis paths.  python ct_topo.py <export_dir> <motion_npz> <trace_dir> <hold_s> <bridge_json>"""
import glob, json, os, sys
import mujoco
import numpy as np
ex, mf, td, hold, bj = sys.argv[1:6]
def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
ref = np.load(mf); a0 = yaw(ref["body_quat_w"][0, 0]); o0 = ref["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]]); to_s = lambda p: (np.asarray(p)[..., :2] - o0) @ R.T
J = json.load(open(bj)); P = J["parts"]
names = sorted((k for k in P if k.startswith("post")), key=lambda k: -(P[k]["center"][2] + P[k]["half_height"]))
posts = np.array([P[k]["center"][:2] for k in names])
def sig(q):
    out = []
    for p in posts:
        d = np.linalg.norm(q - p, axis=1); k = int(np.argmin(d)); k0, k1 = max(k - 3, 0), min(k + 3, len(q) - 1)
        v = q[k1] - q[k0]; w = p - q[k]
        out.append((k, "L" if v[0] * w[1] - v[1] * w[0] > 0 else "R", float(d[k])))
    order = "".join(str(i + 1) for i in np.argsort([o[0] for o in out]))
    return order, "".join(o[1] for o in out), [round(o[2], 3) for o in out]
pr = ref["body_pos_w"][:, 0, :2]
print("posts (real numbering 1 = high), start frame:", [tuple(np.round(to_s(p), 2)) for p in posts], "radius", P[names[0]]["radius"])
print("reference pelvis: order / side of each post (L = post on the left of the travel direction) / min dist:", sig(pr))
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
M = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb")); D = mujoco.MjData(M)
pid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot/" + meta["body_names"][0])
H = int(round(float(hold) / meta["sim"]["control_dt"])); T = len(ref["joint_pos"])
from collections import Counter
cnt = Counter(); md = []
for f in sorted(glob.glob(os.path.join(td, "*.npz"))):
    qp = np.load(f)["qpos"][H:H + T]
    xy = []
    for q in qp[::2]:
        D.qpos[:] = q[:M.nq]; mujoco.mj_kinematics(M, D); xy.append(D.xpos[pid, :2].copy())
    o, s, d = sig(np.array(xy)); cnt[(o, s)] += 1; md.append(d)
print("robot runs:", dict(cnt)); print("robot pelvis min dist per post, 5th pct:", np.round(np.percentile(np.array(md), 5, axis=0), 3))
