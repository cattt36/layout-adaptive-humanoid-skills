"""Failure decomposition of the finished CV students (test clips, seeds given): for each level and action,
fall-or-incomplete = 1 - success; completed but a stake missed = success - success_v2; bridge: share of successes without
any contact.  Condition: randomised (dr1) and nominal (dr0), no push.   python3 cv_fail_decomp.py [seeds]"""
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2"
R = f"{H}/runs"
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1").split(",")]
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]


def rows(F, s, P, dr):
    stem = "v6g10n" + ("" if F == 1 else f"_f{F}")
    f = f"{R}/results_{stem}_s{s}_dr{dr}/eval_student_{P}.json"
    return {r["clip"]: r for r in json.load(open(f))["clips"]} if os.path.exists(f) else None


folds = [F for F in (1, 2, 3, 4) if all(rows(F, s, P, "1") for s in SEEDS for P in LV)]
tests = {F: json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))["test"] for F in folds}
print("folds", folds, "seeds", SEEDS)
for dr, name in (("1", "randomised"), ("0", "nominal")):
    print(f"\n== {name}, no push (% of rollouts on test clips)")
    for T in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
        for P in LV:
            v = [rows(F, s, P, dr)[c] for F in folds for s in SEEDS for c in tests[F] if c.startswith(T)]
            m = lambda k: 100 * sum((r.get(k) or 0) for r in v) / len(v)  # noqa: E731
            if T.startswith(("ThreeStepPoleRun", "FiveStepPoleRun")):
                print(f"  {T:12s} L{P[1]}: success {m('success_v2'):5.1f} | fell/incomplete {100 - m('success'):5.1f} | completed but missed a stake "
                      f"{m('success') - m('success_v2'):5.1f} | all stakes stepped {m('stakes_all_stepped'):5.1f}")
            else:
                print(f"  {T:12s} L{P[1]}: success {m('success_v2'):5.1f} | fell/incomplete {100 - m('success'):5.1f} | success without any bridge contact "
                      f"{m('success_no_contact'):5.1f} | body err {m('body_err') / 100 * 100:5.3f} m")
