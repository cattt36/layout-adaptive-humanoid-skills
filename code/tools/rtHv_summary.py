"""acceptance tables of the stakes student with the mocap velocity (tools/rtH_vel.sh).
  python rtHv_summary.py <out_rtHv> <IT>"""
import glob
import json
import os
import sys

import numpy as np

O, IT = sys.argv[1], sys.argv[2]
C = ["R3", "RP10", "RP10X", "RP10XS"]


def sv(f):
    return f"{100 * np.mean([r['success_v2'] for r in json.load(open(f))['rows']]):.0f}" if os.path.exists(f) else "-"


for M in [f"c{IT}", "c1000"]:
    print(f"## {M} (64 trials)\n\n| layout | " + " | ".join(C) + " |\n|---|" + "---|" * len(C))
    for n in ["D1", "D2", "N3", "S1", "S2", "N1", "T1", "T2", "N2"]:
        print(f"| {n} | " + " | ".join(sv(f"{O}/{M}/{n}_{c}.json") for c in C) + " |")
    print()
print("## stakes candidates (RP10X + stake placement errors, 32 trials)\n\n| layout | candidate | c" + IT + " | c1000 |\n|---|---|---|---|")
for f in sorted(glob.glob(f"{O}/cand_c{IT}/*.json")):
    b = os.path.basename(f)
    lay, cl = b[:-5].split("_", 1)
    print(f"| {lay} | {cl} | {sv(f)} | {sv(f'{O}/cand_c1000/{b}')} |")
