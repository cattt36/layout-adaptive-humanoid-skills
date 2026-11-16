"""Sanity check of the stake-hit criterion on the REFERENCE itself (perfect tracking must hit every stake)."""
import glob, json, os, sys
import numpy as np
import mujoco
root = sys.argv[1]  # data/ref_stakes/<task>
m = mujoco.MjModel  # unused, body order from mjlab G1: ankles are bodies 6 and 12 (1-based world excluded)
names = None
for f in sorted(glob.glob(os.path.join(root, "*_stakes.json")))[:12]:
    meta = json.load(open(f))
    mo = np.load(f.replace("_stakes.json", "_motion.npz"))
    bp = mo["body_pos_w"]  # [T, 30, 3]  order = mjlab body order without world
    L, R = bp[:, 6], bp[:, 12]   # left/right_ankle_roll_link (model body ids 7 / 13, world excluded)
    out = []
    for s in meta["stakes"]:
        foot = L if s["foot"] == "L" else R
        a = int(round(s["stance"][0] * 50 / meta["fps"])) - 1 - 8
        e = int(round(s["stance"][1] * 50 / meta["fps"])) - 1 + 8
        seg = foot[max(a, 0):e + 1]
        dxy = np.hypot(seg[:, 0] - s["x"], seg[:, 1] - s["y"])
        ok = (dxy < 0.18) & (seg[:, 2] > 0.10) & (seg[:, 2] < 0.20)
        out.append((int(ok.sum()), round(float(dxy.min()), 3), round(float(seg[:, 2].min()), 3), round(float(seg[:, 2].max()), 3)))
    print(meta["trial"], "(hits, min dxy, min z, max z) per stake:", out)
