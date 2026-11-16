"""Trigger frames (50 Hz clip frames) of the motion-capture dropout experiment (pre-registered experiment C):
ThreeStepPoleRun / FiveStepPoleRun: 0.5 s before the first stake stance; CrouchedTraverse: 0.5 s before the first frame with a body point under the
beam (within its length, |across| < 0.25 m).  Writes runs/freeze_table.json."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
DIRS = [f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun", f"{H}/ref_stakes_s83_h10/FiveStepPoleRun", f"{H}/ref_bridge_video_lift/CrouchedTraverse"]
tab = {}
for d in DIRS:
    for f in sorted(glob.glob(f"{d}/*_motion.npz")):
        name = os.path.basename(f).replace("_motion.npz", "")
        sj, bj = f.replace("_motion.npz", "_stakes.json"), f.replace("_motion.npz", "_bridge.json")
        if os.path.exists(sj):
            j = json.load(open(sj))
            first = j["stakes"][0]["stance"][0] * 50.0 / j["fps"]
        else:
            m = np.load(os.path.realpath(f))
            p = json.load(open(os.path.realpath(bj)))["parts"]["plank"]
            c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
            dd, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            rel = m["body_pos_w"][:, :, :2] - c[:2]
            inside = (np.abs(rel @ n) < 0.25) & (np.abs(rel @ dd) < half[0])
            first = float(np.where(inside.any(1))[0][0])
        tab[name] = max(0, int(round(first - 25)))
json.dump(tab, open("/root/gpufree-data/hnu_v2/runs/freeze_table.json", "w"), indent=1)
print(len(tab), "clips;", {k: v for k, v in list(tab.items())[:3]}, "...")
