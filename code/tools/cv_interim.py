"""Interim results of the CV students (L40 server): test-clip success (criterion v2, task) per fold, per action, per
condition, L0-L4 + teacher on the same clips; pooled over the finished folds (clip-weighted: every test clip counts once).
python3 cv_interim.py [seeds=1]"""
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2"
R = f"{H}/runs"
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1").split(",")]
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CN = {"CrouchedTraverse": "DuMu", "ThreeStepPoleRun": "W1", "FiveStepPoleRun": "W2"}
CONDS = [("nominal", "0", ""), ("nom+push", "0", "_push"), ("random", "1", ""), ("rand+push", "1", "_push")]


def split_of(F):
    return json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))


def rows(path):
    if not os.path.exists(path):
        return None
    return {r["clip"]: r for r in json.load(open(path))["clips"]}


def student(F, s, P, dr, x, crit=""):
    stem = "v6g10n" + crit + ("" if F == 1 else f"_f{F}")
    return rows(f"{R}/results_{stem}_s{s}_dr{dr}/eval_student_{P}{x}.json")


def teacher(dr, x, crit=""):
    return rows(f"{R}/results_v6g10n{crit}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


folds = [F for F in (1, 2, 3, 4) if all(student(F, s, P, "0", "") for s in SEEDS for P in LV)]
print(f"finished folds for seeds {SEEDS}: {folds}  (L0-L4 complete)")
key = "success_v2"
for scope in [[F] for F in folds] + ([folds] if len(folds) > 1 else []):
    test = [c for F in scope for c in split_of(F)["test"]]
    print(f"\n=== folds {scope}: {len(test)} test clips; success v2 % (mean over clips; each clip 32 rollouts); seeds {SEEDS}")
    for cname, dr, x in CONDS:
        tr = teacher(dr, x)
        line = f"{cname:10s}"
        for T in TASKS:
            cl = [c for c in test if c.startswith(T)]
            vals = []
            for P in LV:
                v = [student(F, s, P, dr, x)[c][key] for s in SEEDS for F in scope for c in cl if c in split_of(F)["test"]]
                vals.append(100 * mean(v))
            t = 100 * mean([tr[c][key] for c in cl])
            line += f" | {CN[T]:4s} " + " ".join(f"{v:5.1f}" for v in vals) + f"  T{t:5.1f}"
        print(line)
    # all actions together (clip-weighted), nominal and randomised
    for cname, dr, x in CONDS:
        tr = teacher(dr, x)
        vals = [100 * mean([student(F, s, P, dr, x)[c][key] for s in SEEDS for F in scope for c in split_of(F)["test"]]) for P in LV]
        print(f"  ALL {cname:10s} " + " ".join(f"L{i} {v:5.1f}" for i, v in enumerate(vals)) + f"   teacher {100 * mean([tr[c][key] for c in test]):5.1f}")
print("\ncolumns per action: L0 L1 L2 L3 L4  T=teacher on the same clips")
