"""DEMO: placement figure of one demo tracker in the start frame (origin = the robot's floor mark, +x = facing the
obstacles): reference pelvis path, simulated robot pelvis paths (tools/prop_spread.py <out>_xy.npz), reference footholds
and simulated landings, stakes (stakes json, 0.40 m) / bridge posts + beam (bridge json).
  python prop_figure.py <spread_prefix> <motion_npz> <png> [props_json]"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


pre, mfile, png = sys.argv[1:4]
pj = sys.argv[4] if len(sys.argv) > 4 else None
a = np.load(pre + "_xy.npz")
m = np.load(mfile)
a0 = yaw(m["body_quat_w"][0, 0])
o0 = m["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p)[..., :2] - o0) @ R.T  # noqa: E731
fig, ax = plt.subplots(figsize=(11, 5.5))
for p in a["robot_pelvis"]:
    ax.plot(p[:, 0], p[:, 1], color="0.75", lw=0.6, zorder=1)
ax.plot(a["ref_pelvis"][:, 0], a["ref_pelvis"][:, 1], "k-", lw=1.6, label="reference pelvis", zorder=3)
col = ["tab:blue", "tab:red"]
for j, (f, s) in enumerate(zip(a["ref_feet"], a["feet_side"])):
    rf = a["robot_feet"][:, j]
    ax.scatter(rf[:, 0], rf[:, 1], s=4, color=col[s], alpha=0.35, zorder=2)
    ax.plot(f[0], f[1], "o", mfc="white", mec=col[s], ms=6, zorder=4)
    ax.annotate(f"{j + 1}{'LR'[s]}", f, xytext=(3, 3), textcoords="offset points", fontsize=7, color=col[s])
if pj:
    d = json.load(open(pj))
    if "stakes" in d:
        for k, st in enumerate(d["stakes"]):
            c = to_s([st["x"], st["y"]])
            ax.add_patch(plt.Circle(c, 0.20, fill=False, ec="tab:green", lw=1.5, zorder=5))
            ax.annotate(f"stake {k + 1}", c, xytext=(0, -14), textcoords="offset points", fontsize=8, ha="center", color="tab:green")
    if "parts" in d:
        for n, p in d["parts"].items():
            c = to_s(p["center"])
            if n.startswith("post"):
                ax.add_patch(plt.Circle(c, p["radius"], color="tab:brown", zorder=5))
                ax.annotate(n, c, xytext=(4, -10), textcoords="offset points", fontsize=8, color="tab:brown")
            elif n == "plank":
                h = p["half"][0]
                yw = p["yaw"] - a0
                dxy = np.array([np.cos(yw), np.sin(yw)]) * h
                ax.plot([c[0] - dxy[0], c[0] + dxy[0]], [c[1] - dxy[1], c[1] + dxy[1]], color="tab:brown", lw=3, alpha=0.5,
                        label=f"beam (top {p['center'][2] + p['half'][2]:.2f} m)", zorder=4)
import os  # noqa: E402
if os.environ.get("DEMO_BEAM"):  # suggested two-post bridge (tools/beam_plan.py): "x1,y1,x2,y2[,underside]"
    v = [float(s) for s in os.environ["DEMO_BEAM"].split(",")]
    ax.plot([v[0], v[2]], [v[1], v[3]], color="tab:purple", lw=4, alpha=0.6, zorder=6,
            label="suggested beam" + (f" (underside >= {v[4]:.2f} m)" if len(v) > 4 else ""))
    for x_, y_ in ((v[0], v[1]), (v[2], v[3])):
        ax.add_patch(plt.Circle((x_, y_), 0.05, color="tab:purple", zorder=7))
ax.plot(0, 0, "k^", ms=10)
ax.annotate("start mark (face +x)", (0, 0), xytext=(5, -15), textcoords="offset points", fontsize=8)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.set_xlabel("x (m), robot start heading")
ax.set_ylabel("y (m), left")
ok = a["ok"]
ax.set_title(f"{pre.split('/')[-1]}: grey = {len(ok)} simulated runs ({100 * ok.mean():.0f} % ok), dots = landings, circles = reference footholds")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(png, dpi=130)
print(png)
