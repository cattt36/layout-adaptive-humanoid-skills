"""Why is criterion-v2 success low on test?  Teacher vs students (v6g10m seed 1, nominal): train / val / test success
(old and v2), and per-trial test values for W1 and CrouchedTraverse."""
import json
import os

H = "/root/gpufree-data/hnu_v2/runs"
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
src = [("teacher", f"{H}/results_v6g10mv2_bank_teacher_dr0/eval_teacher_teacher.json")] + \
      [(f"L{P[1]}", f"{H}/results_v6g10mv2_s1_dr0/eval_student_{P}.json") for P in ("Z0", "Z1", "Z2", "Z3", "Z4")]
print("success % (old -> v2)  DuMu / ThreeStepPoleRun / FiveStepPoleRun")
for name, f in src:
    s = json.load(open(f))["summary"]
    cells = []
    for sp in ("train", "val", "test"):
        cells.append(f"{sp}: " + " ".join(f"{100 * s[f'{t}/{sp}']['success']:.0f}->{100 * s[f'{t}/{sp}']['success_v2']:.0f}" for t in T))
    print(f"{name:8s} " + " | ".join(cells))
print("\nper test trial, v2 success %  (teacher | L0 L1 L2 L3 L4)")
rows = {name: {c["clip"]: c for c in json.load(open(f))["clips"]} for name, f in src}
for clip in sorted(c for c, r in rows["teacher"].items() if r["split"] == "test" and not c.startswith("FiveStepPoleRun")):
    vals = [rows[n][clip]["success_v2"] for n, _ in src]
    olds = [rows[n][clip]["success"] for n, _ in src]
    print(f"{clip:16s} v2 {100 * vals[0]:3.0f} | " + " ".join(f"{100 * v:3.0f}" for v in vals[1:]) +
          "    (old: " + " ".join(f"{100 * v:3.0f}" for v in olds) + ")")
