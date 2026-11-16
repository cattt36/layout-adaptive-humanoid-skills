"""DEMO: time profile of a standard-start clip in its start frame (origin = frame-0 pelvis, +x = frame-0
heading): every 0.5 s t, yaw (deg), pelvis x / y / z; stake stance windows (s) or bridge posts in the start frame.
  python clip_profile.py <motion_dir> <clip> [...]"""
import json
import os
import sys

import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


d = sys.argv[1]
for clip in sys.argv[2:]:
    m = np.load(os.path.join(d, clip + "_motion.npz"))
    fps = float(m["fps"])
    q = m["body_quat_w"][:, 0]
    p = m["body_pos_w"][:, 0]
    h = np.unwrap(yaw(q))
    h0 = h[0]
    R = np.array([[np.cos(-h0), -np.sin(-h0)], [np.sin(-h0), np.cos(-h0)]])
    xy = (p[:, :2] - p[0, :2]) @ R.T
    print(f"== {clip}  {len(h) / fps:.1f} s")
    j = os.path.join(d, clip + "_stakes.json")
    if os.path.exists(j):
        s = json.load(open(j))
        for k, st in enumerate(s["stakes"]):
            c = (np.array([st["x"], st["y"]]) - p[0, :2]) @ R.T
            print(f"   stake {k + 1} {st['foot']} at ({c[0]:.2f}, {c[1]:.2f}) stance {st['stance'][0] / s['fps']:.2f}-{st['stance'][1] / s['fps']:.2f} s")
    j = os.path.join(d, clip + "_bridge.json")
    if os.path.exists(j):
        b = json.load(open(j))
        print("   bridge keys", list(b.keys())[:12])
    for k in range(0, len(h), int(fps / 2)):
        print(f"   t {k / fps:4.1f}  yaw {np.degrees(h[k] - h0):6.0f}  x {xy[k, 0]:5.2f}  y {xy[k, 1]:5.2f}  z {p[k, 2]:4.2f}")
