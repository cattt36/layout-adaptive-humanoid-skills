"""DEMO: placement sheet for a whole-bridge layout (tools/beam4.py DEMO_RIGID=1 cand json): post centres in the
start frame (origin O = floor mark under the pelvis at the start, +x = start heading), tape distances to O and to the
second mark A = (A_X, 0), and a top view with the robot pelvis paths of the sim runs (sim2sim HNU_S2S_TRACE npz).
  python bridge_place.py <export_dir> <cand_json> <motion_npz> <trace_dir> <hold_s> <out_png> [A_X]"""
import glob
import json
import os
import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


ex, cj, mfile, tdir, hold_s, png = sys.argv[1:7]
AX = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
ref = np.load(mfile)
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = ref["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T  # noqa: E731
J = json.load(open(cj))
P = J["parts"]
pl = P["plank"]
yw = float(pl["yaw"]) - a0
ax = np.array([np.cos(yw), np.sin(yw)])
nrm = np.array([-ax[1], ax[0]])
# numbering as on the real bridge (mocap D2 table): post 1 = highest top, post 4 = lowest
names = sorted((k for k in P if k.startswith("post")), key=lambda k: -(P[k]["center"][2] + P[k]["half_height"]))
ps = [to_s(P[k]["center"]) for k in names]
A = np.array([AX, 0.0])
print("| 柱 | x (m) | y (m) | 到 O 的距离 (m) | 到 A 的距离 (m) | 柱顶 = 梁下沿高度 (m) |\n|---|---|---|---|---|---|")
for i, p in enumerate(ps):
    k = names[i]
    print(f"| {i + 1} | {p[0]:.2f} | {p[1]:.2f} | **{np.linalg.norm(p):.2f}** | **{np.linalg.norm(p - A):.2f}** | {P[k]['center'][2] + P[k]['half_height']:.3f} |")
print(f"\n柱 1 → 柱 4 中心距离 {np.linalg.norm(ps[-1] - ps[0]):.2f} m（核对用）；相邻柱中心距 "
      + " / ".join(f"{np.linalg.norm(ps[i + 1] - ps[i]):.2f}" for i in range(len(ps) - 1)) + " m")
print(f"梁方向与机器人起步方向（+x）夹角 {np.degrees(np.arctan2(ax[1], ax[0])) % 360:.0f}°（逆时针为正）；梁是斜的，柱 1 → 柱 4 "
      f"柱顶 {P[names[0]]['center'][2] + P[names[0]]['half_height']:.3f} → {P[names[-1]]['center'][2] + P[names[-1]]['half_height']:.3f} m（按 动捕实测的桥高）")
# top view
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + meta["body_names"][0])
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
fig, axp = plt.subplots(figsize=(8, 8))
for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
    qp = np.load(f)["qpos"][H:]
    xy = []
    for q in qp[::4]:
        data.qpos[:] = q[:model.nq]
        mujoco.mj_kinematics(model, data)
        xy.append(data.xpos[pid, :2].copy())
    xy = to_s(np.array(xy))
    axp.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=0.5, alpha=0.3)
L = float(pl["half"][0])
c = to_s(pl["center"])
e = np.array([c - L * ax, c + L * ax])
axp.plot(e[:, 0], e[:, 1], color="saddlebrown", lw=6, alpha=0.5, solid_capstyle="butt", label="beam")
for k, p in zip(names, ps):
    b = P.get("base" + k[4:])
    if b:
        h = np.array(b["half"][:2])
        by = float(b["yaw"]) - a0
        u, v = np.array([np.cos(by), np.sin(by)]), np.array([-np.sin(by), np.cos(by)])
        cc = to_s(b["center"])
        box = np.array([cc + sx * h[0] * u + sy * h[1] * v for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))])
        axp.plot(box[:, 0], box[:, 1], color="green", lw=1.5)
    axp.add_patch(plt.Circle(p, float(P[k]["radius"]), color="k"))
for i, p in enumerate(ps):
    axp.annotate(f"post {i + 1} h{P[names[i]]['center'][2] + P[names[i]]['half_height']:.2f} ({p[0]:.2f}, {p[1]:.2f})", p, xytext=(8, 8), textcoords="offset points", fontsize=9)
axp.plot(0, 0, "r^", ms=10)
axp.annotate("O start mark (0, 0), face +x", (0, 0), xytext=(8, -14), textcoords="offset points", color="r")
axp.plot(*A, "rs", ms=8)
axp.annotate(f"A ({AX:.0f}, 0)", A, xytext=(8, -14), textcoords="offset points", color="r")
axp.annotate("", xy=(0.8, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=2))
axp.set_aspect("equal")
axp.grid(alpha=0.3)
axp.set_xlabel("x (m), robot start heading")
axp.set_ylabel("y (m), left positive")
axp.set_title(os.path.basename(cj))
fig.savefig(png, dpi=130, bbox_inches="tight")
print(png)
