"""waist-offset acceptance table (tools/rtH_waist.sh).  python rtHw_summary.py <out_rtHw> <IT>"""
import json
import os
import sys

import numpy as np

O, IT = sys.argv[1], sys.argv[2]
W = [("0_0", "waist 0"), ("0_0.10", "pitch +0.10"), ("0_0.20", "pitch +0.20"), ("-0.10_0.10", "roll -0.10 pitch +0.10"),
     ("0_-0.10", "pitch -0.10")]


def sv(f):
    return f"{100 * np.mean([r['success_v2'] for r in json.load(open(f))['rows']]):.0f}" if os.path.exists(f) else "-"


print("RP10X (rope 10 s + real-robot errors), stakes + placement errors; waist roll / pitch stuck at the offset (rad); 64 trials\n")
print("| layout | model | " + " | ".join(w[1] for w in W) + " |\n|---|---|" + "---|" * len(W))
for n in ["D1", "D2", "N3", "S1", "S2", "N1", "T1", "T2", "N2"]:
    ms = [f"rtHwb_s5_c{IT}", "rtGm_s5_c1000"] if n in ("D1", "D2", "N3") else [f"rtHws_s5_c{IT}", "rtHv_s5_c2000"]
    for M in ms:
        print(f"| {n} | {M.split('_s5')[0]} | " + " | ".join(sv(f"{O}/{M}/{n}_w{w[0]}.json") for w in W) + " |")
