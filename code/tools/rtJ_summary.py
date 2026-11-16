"""acceptance tables of the standard-start students (tools/rtJ_std.sh, rtJw.sh).
  python rtJ_summary.py <out_dir> <IT> [waist]"""
import json
import os
import sys

import numpy as np

O, IT = sys.argv[1], sys.argv[2]
waist = len(sys.argv) > 3 and sys.argv[3] == "waist"
if waist:
    C, tags = ["W0_0", "W0_0.30", "W0.30_0.40"], ("rtJwb", "rtJws")
    print("RP10X (+ stake placement errors for the stakes), waist stuck at (roll, pitch) rad; success v2 % (median drift while "
          "standing: position / heading), 64 trials, pendulum rope\n")
else:
    C, tags = ["R3", "RP10", "RP10X", "RP10XS"], ("rtJb", "rtJs")
    print("success v2 % (median drift while standing: position / heading), 64 trials, pendulum rope\n")


def cell(f):
    if not os.path.exists(f):
        return "-"
    rows = json.load(open(f))["rows"]
    sv = 100 * np.mean([r["success_v2"] for r in rows])
    hd = [r.get("hold_drift") for r in rows if r.get("hold_drift")]
    if hd:
        dm, ym = np.median([h[0] for h in hd]), np.median([abs(h[1]) for h in hd])
        return f"{sv:.0f} ({100 * dm:.0f} cm / {ym:.0f} deg)"
    return f"{sv:.0f}"


for tag, lays in ((tags[0], ["D1", "D2", "N3"]), (tags[1], ["S1", "S2", "N1", "T1", "T2", "N2"])):
    for c_ in (IT, "1000"):
        M = f"{tag}_c{c_}"
        print(f"## {M}\n\n| layout | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
        for n in lays:
            print(f"| {n} | " + " | ".join(cell(f"{O}/{M}/{n}_{c}.json") for c in C) + " |")
        print()
