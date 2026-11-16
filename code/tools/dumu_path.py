"""Pelvis path relative to the beam (across = signed distance to the beam axis, along = position along the beam),
pelvis height, every 0.5 s, for a few CrouchedTraverse trials."""
import json
import os
import sys

import numpy as np

D = "/root/gpufree-data/hnu_v2/data/ref_bridge_video_lift/CrouchedTraverse"
for c in sys.argv[1:]:
    m = np.load(os.path.realpath(f"{D}/{c}_motion.npz"))
    j = json.load(open(f"{D}/{c}_bridge.json"))
    p = j["parts"]["plank"]
    ctr, yaw = np.array(p["center"]), p["yaw"]
    d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    pel = m["body_pos_w"][:, 0]
    rel = pel[:, :2] - ctr[:2]
    posts = sorted(round(float((np.array(v["center"][:2]) - ctr[:2]) @ d), 2) for k, v in j["parts"].items() if k.startswith("post"))
    print(f"{c}: plank half-length {p['half'][0]:.2f}, posts along {posts}, yaw {np.degrees(yaw):.0f} deg, start xy {pel[0, :2].round(2)}")
    print("   t:     " + " ".join(f"{t / 50:5.1f}" for t in range(0, len(pel), 25)))
    print("   across " + " ".join(f"{rel[t] @ n:+5.2f}" for t in range(0, len(pel), 25)))
    print("   along  " + " ".join(f"{rel[t] @ d:+5.2f}" for t in range(0, len(pel), 25)))
    print("   height " + " ".join(f"{pel[t, 2]:5.2f}" for t in range(0, len(pel), 25)))
