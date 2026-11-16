"""Underside/top profile of the DuMuQiao plank (Line001) along its long axis, in lab metres (Y up)."""
import sys, json
import numpy as np
from pxr import Usd, UsdGeom

st = Usd.Stage.Open(sys.argv[1])
out = {}
for name in ("Line001", "Cylinder023", "Cylinder024", "Cylinder025", "Cylinder018", "Cylinder057"):
    p = st.GetPrimAtPath(f"/dumuqiao/{name}")
    pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get()) * 0.01  # cm -> m
    M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
    pts = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
    out[name] = pts
plank = out["Line001"]
z = plank[:, 2]
bins = np.linspace(z.min(), z.max(), 27)
prof = []
for a, b in zip(bins[:-1], bins[1:]):
    m = (z >= a) & (z < b)
    if m.any():
        prof.append([round(float((a + b) / 2), 3), round(float(plank[m, 1].min()), 3), round(float(plank[m, 1].max()), 3),
                     round(float(plank[m, 0].min()), 3), round(float(plank[m, 0].max()), 3)])
print("z_mid, y_min, y_max, x_min, x_max")
for r in prof:
    print(r)
posts = {k: [round(float(v[:, 0].mean()), 3), round(float(v[:, 2].mean()), 3), round(float(v[:, 1].max()), 3)]
         for k, v in out.items() if k != "Line001"}
print("posts (x, z, top):", posts)
json.dump(dict(profile=prof, posts=posts), open(sys.argv[2], "w"), indent=1)
