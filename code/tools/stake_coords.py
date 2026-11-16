"""DEMO: stake coordinates for the stakes-on trackers, start frame (origin = floor mark, +x = start heading):
reference positions and the sim-calibrated starting positions (tools/calib_stakes.py: moved to the robot's mean landing).
  python stake_coords.py <motion_npz> <ref_stakes_json> <calibrated_stakes_json>"""
import json
import sys

import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


mfile, rj, cj = sys.argv[1:4]
m = np.load(mfile)
a0 = yaw(m["body_quat_w"][0, 0])
o0 = m["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float) - o0) @ R.T  # noqa: E731
r, c = json.load(open(rj)), json.load(open(cj))
print("| 桩 | 首轮摆放 x (m) | 首轮摆放 y (m) | 参考位置 x | 参考位置 y |\n|---|---|---|---|---|")
for k, (a, b) in enumerate(zip(r["stakes"], c["stakes"])):
    pa, pb = to_s([a["x"], a["y"]]), to_s([b["x"], b["y"]])
    print(f"| {k + 1} | **{pb[0]:.2f}** | **{pb[1]:.2f}** | {pa[0]:.2f} | {pa[1]:.2f} |")
print("\n桩：直径 0.40 m，高 0.10 m，坐标为桩顶圆心。首轮位置 = 仿真里机器人在参考位置的桩上跑 64 次的平均落点；"
      "真机以实测落点为准再挪（README 6.5）。")
