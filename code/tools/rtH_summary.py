"""acceptance table of the LiDAR students (tools/rtH_lidar.sh).
  python rtH_summary.py <out_rtHl> <out_rtGm/lio> <IT>"""
import json
import os
import sys

import numpy as np

O, OL, IT = sys.argv[1], sys.argv[2], sys.argv[3]
C = ["MOCAP", "L1", "P50", "P10a", "P10b", "P10aS"]


def sv(f):
    return f"{100 * np.mean([r['success_v2'] for r in json.load(open(f))['rows']]):.0f}" if os.path.exists(f) else "-"


print("MOCAP = mocap pose (RP10X); L1 = map-based LiDAR pose (2 cm, 1 % drift) at 50 Hz without delay; P50 = + 30 ms; "
      "P10a / P10b = 10 Hz scans + 50 / 100 ms; P10aS = P10a + stake / bridge placement errors 3 cm + perception noise\n")
for M in [f"rtHl_s5_c{IT}", "rtHl_s5_c1000", "rtGm_s5_c1000"]:
    print(f"## {M} (64 trials)\n\n| layout | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
    for n in ["D1", "D2", "N3", "S1", "S2", "N1", "T1", "T2", "N2"]:
        row = [sv(f"{O}/{M}/{n}_{c}.json") if os.path.exists(f"{O}/{M}/{n}_{c}.json") else sv(f"{OL}/{n}_{c}.json") for c in C]
        print(f"| {n} | " + " | ".join(row) + " |")
    print()
