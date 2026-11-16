"""Print prims, types, transforms and extents of a USD scene."""
import sys
from pxr import Usd, UsdGeom

for path in sys.argv[1:]:
    st = Usd.Stage.Open(path)
    print("=====", path, "upAxis", UsdGeom.GetStageUpAxis(st), "metersPerUnit", UsdGeom.GetStageMetersPerUnit(st))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for p in st.Traverse():
        line = f"{p.GetPath()}  [{p.GetTypeName()}]"
        if p.IsA(UsdGeom.Xformable):
            m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            line += f"  pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})"
        if p.IsA(UsdGeom.Boundable):
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()
            line += f"  bbox=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})..({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})"
        for a in ("size", "radius", "height", "extent"):
            at = p.GetAttribute(a)
            if at and at.HasValue():
                line += f"  {a}={at.Get()}"
        print(line)
