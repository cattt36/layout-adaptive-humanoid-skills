"""Students v6g10s (sync sensor-only latency) vs old v6g10 vs teacher bank: success / body error per task and setting."""
import glob
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2/runs"
SEEDS = sys.argv[1:] or ["1"]


def get(f, T):
    if not os.path.exists(f):
        return None
    x = json.load(open(f))["summary"][T + "/all"]
    return 100 * x["success"], 100 * x["body_err"]


def fmt(v):
    return "    -     " if v is None else f"{v[0]:5.1f}/{v[1]:3.1f}"


for T in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    print(f"== {T}   (success % / body err cm)   nominal | nominal+push | randomised | randomised+push")
    tb = [get(f"{H}/results_v6g10s_bank_teacher_dr{d}/eval_teacher_teacher{x}.json", T) for d in "01" for x in ("", "_push")]
    print("  teacher    ", " | ".join(fmt(v) for v in tb))
    for P in ("Z0", "Z1", "Z2", "Z3", "Z4"):
        for s in SEEDS:
            new = [get(f"{H}/results_v6g10s_s{s}_dr{d}/eval_student_{P}{x}.json", T) for d in "01" for x in ("", "_push")]
            if all(v is None for v in new):
                continue
            old = [get(f"{H}/results_v6g10_s{s}_dr{d}/eval_student_{P}{x}.json", T) for d in "01" for x in ("", "_push")]
            print(f"  L{P[1]} s{s} new ", " | ".join(fmt(v) for v in new))
            if any(v is not None for v in old):
                print(f"  L{P[1]} s{s} old ", " | ".join(fmt(v) for v in old))
