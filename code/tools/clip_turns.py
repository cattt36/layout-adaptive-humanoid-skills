"""DEMO: how much a standard-start clip turns, in its start frame (origin = frame-0 pelvis, +x = frame-0
heading).  Phases: approach (start -> first crouch z < 0.66 m for CT / first stake stance for stakes), skill (-> exit:
CT pelvis back above 0.70 m, stakes last stance end), tail (-> end).  Per phase: max |yaw| (deg); also the path
direction start -> exit (deg) and the time of the first |yaw| > 90 deg (the human's end turn-around).
  python clip_turns.py <motion_dir> [...]"""
import glob
import json
import os
import sys

import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*_motion.npz"))):
        m = np.load(f)
        fps = float(m["fps"])
        h = np.degrees(np.unwrap(yaw(m["body_quat_w"][:, 0])))
        h -= h[0]
        p = m["body_pos_w"][:, 0]
        a = np.radians(np.degrees(np.unwrap(yaw(m["body_quat_w"][:, 0])))[0])
        R = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
        xy = (p[:, :2] - p[0, :2]) @ R.T
        js = f.replace("_motion.npz", "_stakes.json")
        if os.path.exists(js):
            s = json.load(open(js))
            k0 = int(s["stakes"][0]["stance"][0] / s["fps"] * fps)
            k1 = int(s["stakes"][-1]["stance"][1] / s["fps"] * fps)
        else:
            low = np.where(p[:, 2] < 0.66)[0]
            k0 = int(low[0]) if len(low) else len(h) // 2
            up = np.where((np.arange(len(h)) > low[-1]) & (p[:, 2] > 0.70))[0] if len(low) else []
            k1 = int(up[0]) if len(up) else len(h) - 1
        k1 = min(k1, len(h) - 1)
        big = np.where(np.abs(h) > 90)[0]
        t90 = f"{big[0] / fps:4.1f}" if len(big) else "  - "
        path = np.degrees(np.arctan2(xy[k1, 1], xy[k1, 0]))
        print(f"{os.path.basename(f)[:-11]:24s} T {len(h) / fps:4.1f}  approach 0-{k0 / fps:3.1f}s max|yaw| {np.abs(h[:k0 + 1]).max():4.0f}"
              f"  skill -{k1 / fps:3.1f}s max|yaw| {np.abs(h[k0:k1 + 1]).max():4.0f}  path dir {path:5.0f}  "
              f"|yaw|>90 at {t90} s  tail max|yaw| {np.abs(h[k1:]).max():4.0f}")
