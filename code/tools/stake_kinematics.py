"""Stake spacing / timing per clip (robot frame) and the flat expert's foot error at the stake windows."""
import glob
import json
import os
import sys

import numpy as np

for T in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    gaps, dts, speeds = [], [], []
    for f in sorted(glob.glob(os.path.join(sys.argv[1], T, "*_stakes.json"))):
        m = json.load(open(f))
        s = m["stakes"]
        for a, b in zip(s[:-1], s[1:]):
            d = np.hypot(b["x"] - a["x"], b["y"] - a["y"])
            dt = (np.mean(b["stance"]) - np.mean(a["stance"])) / m["fps"]
            gaps.append(d); dts.append(dt); speeds.append(d / dt)
    print(f"{T}: stake gap {np.median(gaps):.2f} m [{np.min(gaps):.2f}, {np.max(gaps):.2f}], "
          f"time between stances {np.median(dts):.2f} s, step speed {np.median(speeds):.2f} m/s [{np.min(speeds):.2f}, {np.max(speeds):.2f}]")
