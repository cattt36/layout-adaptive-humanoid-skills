"""Compare BVH toe positions with the CSV force-plate toe marks (frame alignment and axes), and print a coarse
time series of the head relative to the marker-defined plank line for one trial."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from human_bridge_clearance import FU, RU, fk, parse_bvh  # noqa: E402

bvh, csvrow = sys.argv[1], sys.argv[2:]
joints, data, dt = parse_bvh(bvh)
P = {k: v / 100.0 for k, v in fk(joints, data).items()}
print("frames", len(data), "dt", dt, "joints", [k for k in P][:60])
for s in csvrow:  # foot,frame,x,z
    foot, fr, x, z = s.split(",")
    fr = int(fr)
    k = "LeftToe" if foot == "L" else "RightToe"
    for off in (-1, 0):
        p = P[k][fr + off]
        print(f"{foot} csv frame {fr} csv (x,z)=({x},{z})  bvh[{fr + off}] {k} = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
H = P["HeadEnd"] if "HeadEnd" in P else P["Head"]
for t in range(0, len(H), 40):
    u = (H[t, 2] - RU[2]) / (FU[2] - RU[2])
    lx = RU[0] + u * (FU[0] - RU[0]); top = RU[1] + u * (FU[1] - RU[1])
    print(f"t {t * dt:5.2f}s head ({H[t, 0]:+.2f}, {H[t, 1]:.2f}, {H[t, 2]:+.2f})  dx {H[t, 0] - lx:+.2f}  plank top {top:.2f}  u {u:.2f}")
