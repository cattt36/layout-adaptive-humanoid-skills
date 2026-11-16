"""Nominal-gap fix tests: original v6g10 students vs round 3 with sync latency vs round 3 with sync sensor-only latency."""
import json
import os

H = "/root/gpufree-data/hnu_v2/runs"
A, B = f"{H}/test_sync_ft", f"{H}/test_sync_ft_sens"


def s(d, P, T):
    f = f"{d}/eval_student_{P}.json"
    if not os.path.exists(f):
        return "  -  "
    return f"{100 * json.load(open(f))['summary'][T + '/all']['success']:5.1f}"


print("task P | orig: nom | sync: nom full | sens: nom lag1 lag01 full | orig under sens latency: lag1 full")
for T in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    for P in ("Z0", "Z3"):
        print(f"{T[:6]} {P} | {s(H + '/results_v6g10_s1_dr0', P, T)} | {s(A + '/eval_nom', P, T)} {s(A + '/eval_full', P, T)} | "
              f"{s(B + '/eval_nom', P, T)} {s(B + '/eval_lag1', P, T)} {s(B + '/eval_lag01', P, T)} {s(B + '/eval_full', P, T)} | "
              f"{s(B + '/old_lag1', P, T)} {s(B + '/old_full', P, T)}")
