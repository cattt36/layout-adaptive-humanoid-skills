"""sim2sim preview (CPU MuJoCo, explicit PD, latency 0-20 ms) of the fold-1 students: success criterion v2 on the
fold-1 TEST clips and on all clips, per level and action.   python s2s_v2_summary.py <export_root>"""
import glob
import json
import os
import sys

import numpy as np

E = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/hnu_v2/deploy_exports/v6g10n_f1_s1"
test = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_cv/fold1/split.json"))["test"])
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
for tag, name in (("sim2sim_0-20_dt0.005.json", "latency 0-20 ms"), ("sim2sim_0-20_dr_dt0.005.json", "latency 0-20 ms + dynamics DR")):
    print(f"\n### {name}: success v2 % on fold-1 TEST clips (all clips in brackets); 4 trials per clip")
    print("| level | " + " | ".join(TASKS) + " | all test |")
    print("|---" * 5 + "|")
    for L in sorted(glob.glob(f"{E}/L?")):
        f = f"{L}/{tag}"
        if not os.path.exists(f):
            continue
        rows = json.load(open(f))["rows"]
        cells = []
        for T in TASKS:
            t = [r["success_v2"] for r in rows if r["task"] == T and r["clip"] in test]
            a = [r["success_v2"] for r in rows if r["task"] == T]
            cells.append(f"{100 * np.mean(t):.1f} ({100 * np.mean(a):.1f})")
        allt = [r["success_v2"] for r in rows if r["clip"] in test]
        print(f"| {os.path.basename(L)} | " + " | ".join(cells) + f" | {100 * np.mean(allt):.1f} |")
