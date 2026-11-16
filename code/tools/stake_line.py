"""How far are the stakes of every trial from a straight line?  Per trial: perpendicular offset of each stake from
the best-fit line (PCA of the stake centres), spacing; W1 = 3 stakes, W2 = 5 stakes (W2 is a zig-zag by design)."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
for name, d in (("W1", f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun"), ("W2", f"{H}/ref_stakes_s83_h10/FiveStepPoleRun")):
    devs = []
    print(f"== {name}")
    for f in sorted(glob.glob(f"{d}/*_stakes.json")):
        j = json.load(open(f))
        xy = np.array([[s["x"], s["y"]] for s in j["stakes"]])
        c = xy.mean(0)
        u = np.linalg.svd(xy - c)[2][0]
        n = np.array([-u[1], u[0]])
        off = (xy - c) @ n
        gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        devs.append(np.abs(off).max())
        print(f"   {os.path.basename(f)[:15]}  feet {''.join(s['foot'] for s in j['stakes'])}  offsets from line "
              f"{' '.join(f'{v:+.2f}' for v in off)} m   gaps {' '.join(f'{g:.2f}' for g in gaps)} m")
    print(f"   max |offset| per trial: median {np.median(devs):.2f} m, max {np.max(devs):.2f} m")
