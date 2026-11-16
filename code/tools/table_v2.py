"""Old (task) vs new (v2) success criterion: teacher bank (all / train trials) and v6g10m students (test / val),
per action DuMu / ThreeStepPoleRun / FiveStepPoleRun, four conditions; stake details for the nominal condition."""
import glob
import json
import os
import re

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
R = f"{H}/results_v6g10mv2"
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
C = [("nom", "0", ""), ("nom+push", "0", "_push"), ("rand", "1", ""), ("rand+push", "1", "_push")]


def s3(s, sp, k):
    return "/".join(f"{100 * s[f'{t}/{sp}'][k]:.0f}" if s.get(f"{t}/{sp}", {}).get(k) is not None else "-" for t in T)


def line(name, files, sp):
    out = []
    for c, dr, x in C:
        f = files(dr, x)
        if not os.path.exists(f):
            out.append(f"{c}: -")
            continue
        s = json.load(open(f))["summary"]
        out.append(f"{c}: old {s3(s, sp, 'success'):10s} v2 {s3(s, sp, 'success_v2'):10s}")
    f = files("0", "")
    det = ""
    if os.path.exists(f):
        s = json.load(open(f))["summary"]
        g = lambda t, k: s[f"{t}/{sp}"].get(k)  # noqa: E731
        det = ("  | nominal W1/W2: all stepped " + "/".join(f"{100 * g(t, 'stakes_all_stepped'):.0f}" for t in T[1:]) +
               "  all fully on " + "/".join(f"{100 * g(t, 'stakes_all_full'):.0f}" for t in T[1:]) +
               "  stepped frac " + "/".join(f"{100 * g(t, 'stakes_stepped_frac'):.0f}" for t in T[1:]) +
               "  legacy all-hit " + "/".join(f"{100 * (g(t, 'stakes_all_hit') or 0):.0f}" for t in T[1:]))
    print(f"{name:14s} [{sp}] " + " | ".join(out) + det)


print("success %  DuMu/W1/W2   (old = task criterion, v2 = + every stake stepped on)")
for sp in ("all", "train", "test"):
    line("teacher", lambda dr, x: f"{R}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json", sp)
for sp in ("test", "val"):
    print()
    for d in sorted(glob.glob(f"{R}_s*_dr0")):
        s = re.search(r"_s(\d+)_dr0", d).group(1)
        for P in ["Z0", "Z1", "Z2", "Z3", "Z4", "NL", "REF"]:
            if os.path.exists(f"{d}/eval_student_{P}.json"):
                line(f"L{P[1]} s{s}" if P[0] == "Z" else f"{P} s{s}",
                     lambda dr, x, s=s, P=P: f"{R}_s{s}_dr{dr}/eval_student_{P}{x}.json", sp)
