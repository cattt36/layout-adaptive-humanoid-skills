"""Compare finished v6g10j students with the v6g10t seed-1 students (task criterion), per split and condition."""
import json
import os

H = "/root/gpufree-data/hnu_v2/runs"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CONDS = [("nom", "0", ""), ("nom+push", "0", "_push"), ("rand", "1", ""), ("rand+push", "1", "_push")]


def load(V, s, P, dr, x):
    f = f"{H}/results_{V}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    return json.load(open(f))["summary"] if os.path.exists(f) else None


for V, s in [("v6g10t", 1), ("v6g10j", 1)] + [("v6g10m", k) for k in range(1, 6)]:
    for P in ("Z0", "Z4", "Z1", "Z2", "Z3", "REF", "NL"):
        rows = []
        for c, dr, x in CONDS:
            r = load(V, s, P, dr, x)
            if r is None:
                continue
            cell = " ".join(f"{100 * r[f'{t}/test']['success']:3.0f}" for t in TASKS)
            if c == "nom":
                tr = " ".join(f"{100 * r[f'{t}/train']['success']:3.0f}" for t in TASKS)
                va = " ".join(f"{100 * r[f'{t}/val']['success']:3.0f}" for t in TASKS)
                hit = " ".join(f"{100 * (r[f'{t}/test'].get('stakes_all_hit') or 0):3.0f}" for t in TASKS[1:])
                dist = " ".join(f"{100 * (r[f'{t}/test'].get('stake_centre_dist_m') or 0):4.1f}" for t in TASKS[1:])
                be = " ".join(f"{100 * r[f'{t}/test']['body_err']:4.1f}" for t in TASKS)
                cell += f" [tr {tr} | va {va}] hit {hit} dist {dist} berr {be}"
            rows.append(f"{c}: {cell}")
        if rows:
            print(f"{V} s{s} {P}:  " + "  ||  ".join(rows))
