"""DEMO: 1:1 start footprint template per clip (A3 landscape PDF, print at 100 % / "actual size").
Frame-0 feet of the reference in the start frame (origin O = floor point under the pelvis, +x = pelvis heading): sole
outline = convex hull of the lowest 1 cm of the G1 foot mesh.  Usage: python foot_template.py <scene.mjb> <motion.npz>
<title> <out.pdf>"""
import sys

import matplotlib
import mujoco
import numpy as np
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def hull(p):
    p = sorted(map(tuple, p))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


scene, mfile, title, out = sys.argv[1:5]
M = mujoco.MjModel.from_binary_path(scene)
ref = np.load(mfile)
bp, bq = ref["body_pos_w"][0], ref["body_quat_w"][0]
a0 = yaw(bq[0])
R2 = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p)[..., :2] - bp[0, :2]) @ R2.T  # noqa: E731
feet = {}
for side, idx in (("左脚 L", 6), ("右脚 R", 12)):
    b = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, f"robot/{'left' if idx == 6 else 'right'}_ankle_roll_link")
    g = [i for i in range(M.ngeom) if M.geom_bodyid[i] == b and M.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH][0]
    mid = M.geom_dataid[g]
    v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]].astype(float)
    Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, M.geom_quat[g]); v = v @ Rg.reshape(3, 3).T + M.geom_pos[g]
    Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, bq[idx]); w = v @ Rb.reshape(3, 3).T + bp[idx]
    sole = w[v[:, 2] < v[:, 2].min() + 0.01]  # lowest 1 cm in the foot frame (also if the foot is pitched)
    xy = to_s(sole)
    fy = np.degrees(yaw(bq[idx]) - a0)
    feet[side] = (hull(xy), fy, to_s(bp[idx]))
# A3 landscape: page x (420 mm) = robot y (left = page left), page y (297 mm) = robot +x (up)
W, Hh = 0.420, 0.297
fig = plt.figure(figsize=(W / 0.0254, Hh / 0.0254))
ax = fig.add_axes([0, 0, 1, 1])
allx = np.concatenate([f[0][:, 0] for f in feet.values()])
xc = 0.5 * (allx.min() + allx.max())
X0 = xc - 0.13  # robot x at the page bottom
ax.set_xlim(W / 2, -W / 2)  # robot +y to the page left
ax.set_ylim(X0, X0 + Hh)
ax.set_aspect("equal")
ax.axis("off")
for side, (h, fy, ank) in feet.items():
    hh = np.vstack([h, h[:1]])
    ax.fill(hh[:, 1], hh[:, 0], color="#d9e8f5", ec="#1f4e79", lw=1.5)
    ax.plot(ank[1], ank[0], "+", color="#1f4e79", ms=12)
    straight = abs(fy) < 1.5
    txt = f"{side}\n{'朝 +x（直脚）' if straight else f'外八 {abs(fy):.1f}°'}"
    ax.text(ank[1], h[:, 0].max() + 0.012, txt, ha="center", va="bottom", fontsize=13,
            color="#1f4e79" if straight else "#b03a2e")
# x line through O over the whole page + y tick line
ax.plot([0, 0], [X0, X0 + Hh], color="r", lw=1.2)
ax.annotate("", xy=(0, X0 + Hh - 0.005), xytext=(0, X0 + Hh - 0.06), arrowprops=dict(arrowstyle="-|>", color="r", lw=2))
ax.text(-0.006, X0 + Hh - 0.03, "+x 朝障碍物（对准地面 x 线）", color="r", fontsize=13, ha="left", va="center")
ax.plot([-0.03, 0.03], [0, 0], color="r", lw=1.2)
ax.plot(0, 0, "o", mfc="none", mec="r", ms=10)
ax.text(0.006, -0.008, "O 骨盆正下方（铅锤点）", color="r", fontsize=12, ha="right", va="top")
for t in np.arange(np.ceil(X0 * 100) / 100, X0 + Hh, 0.01):  # 1 cm ticks on the x line
    L = 0.006 if round(t * 100) % 5 else 0.012
    ax.plot([-L / 2, L / 2], [t, t], color="r", lw=0.6)
# 1:1 check bars
yb = X0 + 0.015
ax.plot([W / 2 - 0.02, W / 2 - 0.12], [yb, yb], color="k", lw=2)
ax.text(W / 2 - 0.07, yb + 0.004, "打印后量这条线 = 10.0 cm（必须 100% 实际大小打印）", fontsize=10, ha="center", va="bottom")
ax.plot([-W / 2 + 0.02, -W / 2 + 0.02], [yb, yb + 0.1], color="k", lw=2)
ax.text(-W / 2 + 0.025, yb + 0.05, "10.0 cm", fontsize=10, ha="right", va="center", rotation=90)
ax.text(0, X0 + 0.005, f"{title}  起点脚印模板 1:1（A3 横向）", fontsize=12, ha="center", va="bottom")
ax.text(W / 2 - 0.02, X0 + Hh - 0.015,
        "用法：1) 红线对准地面 x 线、O 对准地面 O 点，四角胶带固定\n2) 吊绳上把两只脚放进轮廓（外八的脚也照轮廓放）\n"
        "3) 模板只用于粗摆（约 ±2°），朝向精确值由 IMU / 激光读出并软件补偿",
        fontsize=10, ha="left", va="top")
fig.savefig(out)
for side, (h, fy, ank) in feet.items():
    print(f"{side}: ankle ({ank[0]:+.3f}, {ank[1]:+.3f}) m, yaw {fy:+.1f} deg, sole x {h[:, 0].min():+.3f}..{h[:, 0].max():+.3f}, y {h[:, 1].min():+.3f}..{h[:, 1].max():+.3f}")
