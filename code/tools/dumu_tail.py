"""CrouchedTraverse: time between the last frame with a body point under the beam and the clip end (the robot must also
stand up and stay up after the bridge), and the reference pelvis height / speed at the clip end."""
import glob
import json
import os

import numpy as np

D = "/root/gpufree-data/hnu_v2/data/ref_bridge_video_lift/CrouchedTraverse"
tails = []
for f in sorted(glob.glob(f"{D}/*_motion.npz")):
    c = os.path.basename(f).replace("_motion.npz", "")
    m = np.load(os.path.realpath(f))
    p = json.load(open(os.path.realpath(f.replace("_motion.npz", "_bridge.json"))))["parts"]["plank"]
    ctr, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - ctr[:2]
    inside = ((np.abs(rel @ n) < 0.25) & (np.abs(rel @ d) < half[0])).any(1)
    last = np.where(inside)[0][-1]
    T = len(inside)
    tail = (T - 1 - last) / 50
    tails.append(tail)
    v = np.linalg.norm(m["body_lin_vel_w"][-1, 0, :2])
    flag = "   <-- < 1 s" if tail < 1.0 else ""
    print(f"{c}  leaves the beam at {last / 50:.2f} s, clip ends {T / 50:.2f} s -> {tail:.2f} s after   "
          f"end pelvis h {m['body_pos_w'][-1, 0, 2]:.2f} m  speed {v:.2f} m/s{flag}")
print(f"tail after the beam: min {min(tails):.2f} s, median {np.median(tails):.2f} s")
