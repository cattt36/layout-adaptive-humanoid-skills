"""DEMO (user: run the robot on the flat floor first, then put the stakes where it lands): move every stake of
a clip by the robot's mean landing offset at that stake (tools/prop_spread.py <prefix>.json, start frame) -> new stakes
json (clip world frame, same format: x / y and stake_xyz_50hz).  The reference's stake-to-ankle relation is kept.
  python calib_stakes.py <motion_npz> <stakes_json> <spread_json> <out_json>"""
import json
import sys

import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


mfile, sj, spj, out = sys.argv[1:5]
m = np.load(mfile)
a0 = yaw(m["body_quat_w"][0, 0])
o0 = m["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
s = json.load(open(sj))
lands = json.load(open(spj))["landings"]
for k, st in enumerate(s["stakes"]):
    c = (np.array([st["x"], st["y"]]) - o0) @ R.T
    best = min(lands, key=lambda l: np.linalg.norm(np.array(l["ref"]) - c))
    d = np.linalg.norm(np.array(best["ref"]) - c)
    off_s = np.array(best["mean"]) if d < 0.30 else np.zeros(2)
    off_w = R.T @ off_s
    st["x"], st["y"] = float(st["x"] + off_w[0]), float(st["y"] + off_w[1])
    s["stake_xyz_50hz"][k][0] = float(s["stake_xyz_50hz"][k][0] + off_w[0])
    s["stake_xyz_50hz"][k][1] = float(s["stake_xyz_50hz"][k][1] + off_w[1])
    print(f"stake {k + 1}: moved {off_s[0]:+.3f} / {off_s[1]:+.3f} m (start frame), landing r95 {best['r95']:.3f} m")
s["calibrated"] = "stakes moved to the simulated mean landings (" + spj + ")"
json.dump(s, open(out, "w"), indent=1)
