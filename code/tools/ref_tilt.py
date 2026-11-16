"""Reference pelvis tilt (angle between the pelvis z axis and the world vertical) per action: is a 1 rad fall threshold
compatible with the deep crouch of CrouchedTraverse?"""
import glob

import numpy as np

H = "/root/gpufree-data/hnu_v2"
DIRS = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}


def tilt(q):  # wxyz -> angle of body z to world z
    w, x, y, z = q.T
    zz = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(zz, -1, 1)))


for act, d in DIRS.items():
    mx, frac = [], []
    for f in sorted(glob.glob(f"{d}/*_motion.npz")):
        t = tilt(np.load(f)["body_quat_w"][:, 0])
        mx.append(t.max())
        frac.append(np.mean(t > 57.3))
    print(f"{act:12s} pelvis tilt max: median {np.median(mx):.0f} deg, max {np.max(mx):.0f} deg; "
          f"share of frames > 57 deg: mean {100 * np.mean(frac):.1f}%  (clips with any > 57 deg: {np.sum(np.array(mx) > 57.3)}/{len(mx)})")
