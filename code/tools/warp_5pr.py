"""DEMO (user: change the stake spacing): narrow the lateral zig-zag of the 5PR reference.  The human hops
0.65-0.85 m sideways between the L and R stakes; the R stakes (2, 4) are pulled towards the line through L stakes 1 and 5
by the factor S (lateral deviation x S).  Feet order L-R-L-R-L, stake heights, timing and the run-up stay as they are.
Like warp_ct.py every frame is shifted RIGIDLY (joints / orientations unchanged, velocities updated): offset constant
during each stake stance (no foot slide), linear during the flights (= ballistic), zero before stake 1 and after stake 5.
The stakes move with their stance offset (the calibrated stakes of std_bvB keep their offset to the reference foot).
  python warp_5pr.py <stakes_root_in> <stakes_root_out> <S>"""
import json
import os
import sys

import numpy as np

src, dst, S = sys.argv[1], sys.argv[2], float(sys.argv[3])
T_ = "FiveStepPoleRun"
mf, sf = (os.path.join(src, T_, f"{T_}_018_{k}") for k in ("motion.npz", "stakes.json"))
d = dict(np.load(mf))
J = json.load(open(sf))
fps = float(d["fps"])
bp = d["body_pos_w"].astype(np.float64)
T = len(bp)
fc = d["foot_contact"].astype(bool)
ANK = {"L": 6, "R": 12}
COL = {"L": 0, "R": 1}
st = J["stakes"]
xy = np.array([[q["x"], q["y"]] for q in st])
# stance of each stake in motion frames: contact block of that foot whose ankle is nearest to the stake
stance = []
for q, p in zip(st, xy):
    c = fc[:, COL[q["foot"]]]
    dist = np.linalg.norm(bp[:, ANK[q["foot"]], :2] - p, axis=1)
    k = int(np.argmin(np.where(c, dist, 9.0)))
    a, b = k, k
    while a > 0 and c[a - 1]:
        a -= 1
    while b < T - 1 and c[b + 1]:
        b += 1
    stance.append((a, b, float(dist[k])))
print("stances (motion frames, ankle-stake dist):", [(a, b, round(e, 3)) for a, b, e in stance])
assert all(stance[i][1] < stance[i + 1][0] for i in range(len(st) - 1)), "stances overlap"
# lateral deviation from the L1-L5 line
ax = (xy[-1] - xy[0]) / np.linalg.norm(xy[-1] - xy[0])
nrm = np.array([-ax[1], ax[0]])
dev = (xy - xy[0]) @ nrm
o_st = ((S - 1.0) * dev)[:, None] * nrm  # [5, 2]
off = np.zeros((T, 2))
for i, (a, b, _) in enumerate(stance):
    off[a:b + 1] = o_st[i]
    if i + 1 < len(stance):
        a2 = stance[i + 1][0]
        w = np.linspace(0, 1, a2 - b + 1)[1:-1, None]
        off[b + 1:a2] = (1 - w) * o_st[i] + w * o_st[i + 1]
off[stance[-1][1] + 1:] = o_st[-1]
off[:stance[0][0]] = o_st[0]
assert np.abs(o_st[0]).max() < 1e-9 and np.abs(o_st[-1]).max() < 1e-9
bp[:, :, :2] += off[:, None]
lv = d["body_lin_vel_w"].astype(np.float64)
lv[:, :, :2] += (np.gradient(off, axis=0) * fps)[:, None]
d["body_pos_w"], d["body_lin_vel_w"] = bp.astype(np.float32), lv.astype(np.float32)
d["start_info"] = np.array(str(d["start_info"]) + f" | 5PR lateral zig-zag x{S} (warp_5pr.py )")
nxy = xy + o_st
for q, p in zip(st, nxy):
    q["x"], q["y"] = float(p[0]), float(p[1])
J["stake_xyz_50hz"] = [[float(p[0]), float(p[1]), z[2]] for p, z in zip(nxy, J["stake_xyz_50hz"])]
J["lateral_scale"] = S
os.makedirs(os.path.join(dst, T_), exist_ok=True)
np.savez(os.path.join(dst, T_, f"{T_}_018_motion.npz"), **d)
json.dump(J, open(os.path.join(dst, T_, f"{T_}_018_stakes.json"), "w"), indent=1)
# layout in the start frame (robot pelvis at frame 0 = O, heading = +x), as in the vK props tables
p0 = bp[0, 0, :2]
w, x, y, z = d["body_quat_w"][0, 0]
yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
for tag, P in (("before", xy), (f"x{S}", nxy)):
    loc = (P - p0) @ R.T
    gaps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    print(f"{tag}: " + " ".join(f"{q['foot']}({u:.2f},{v:.2f})" for q, (u, v) in zip(st, loc)),
          "| centre distances", np.round(gaps, 2).tolist())
print("max |offset|", round(float(np.abs(off).max()), 3), "m ->", dst)
