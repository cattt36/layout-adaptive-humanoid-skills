"""DEMO ONLY: flat-ground version of a standard-start stakes clip (the robot does the stepping-stone motion on
the floor, the props stand beside the footholds).  The raised references (ref_stakes_*_h10) lift the whole body by the
stake height h on a piecewise-linear ramp (stakes json "ramp" = [up0, up1, down0, down1] at the json fps, clip timeline); this undoes
it per frame: body z -= r(src), body z velocity -= dr/dt.  Joints, orientations, contacts unchanged.

  python flatten_stakes.py <in_dir> <clip> <out_dir>
"""
import json
import os
import sys

import numpy as np

in_dir, clip, out_dir = sys.argv[1:4]
d = dict(np.load(os.path.join(in_dir, clip + "_motion.npz")))
s = json.load(open(os.path.join(in_dir, clip + "_stakes.json")))
h = float(s["height"])
u0, u1, d0, d1 = s["ramp"]
# ramp / stance frames are at the json fps (120 Hz csv frames) on the clip's own (standard-start) timeline
src = np.arange(len(d["joint_pos"])) * float(s.get("fps", 120.0)) / float(d["fps"])
r = h * np.clip(np.minimum((src - u0) / max(u1 - u0, 1e-6), (d1 - src) / max(d1 - d0, 1e-6)), 0.0, 1.0)
fps = float(d["fps"])
dr = np.gradient(r) * fps
bp = d["body_pos_w"].copy()
bp[:, :, 2] -= r[:, None].astype(np.float32)
bv = d["body_lin_vel_w"].copy()
bv[:, :, 2] -= dr[:, None].astype(np.float32)
d["body_pos_w"], d["body_lin_vel_w"] = bp, bv
d["start_info"] = np.array(str(d["start_info"]) + " | flattened (demo, stakes removed)")
os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir, clip + "_motion.npz"), **d)
# report: lowest body point per frame (a foot) while fully raised (on the stakes) and elsewhere, before / after
z0 = np.load(os.path.join(in_dir, clip + "_motion.npz"))["body_pos_w"][:, :, 2].min(1)
z1 = bp[:, :, 2].min(1)
on = r > h - 1e-6
print(f"{clip}: h {h}, ramp {s['ramp']}, src {src.min():.0f}-{src.max():.0f}, raised frames {(r > 1e-6).sum()}/{len(r)}; "
      f"lowest body z on stakes: before p10 {np.percentile(z0[on], 10):.3f} after {np.percentile(z1[on], 10):.3f}; "
      f"elsewhere p10 {np.percentile(z1[~on], 10):.3f}; overall min after {z1.min():.3f}")
