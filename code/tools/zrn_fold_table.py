"""Per-fold closed-loop success of ER (results_v6g10xm*) and ER-conv (results_v6g10n*), level ZR, nominal, conditions A
and B, for the per-fold table of the Supplementary Material.  Per fold: mean over the test trials of the
mean over RL seeds 1-3; All: mean over all 76 test trials.
  python zrn_fold_table.py"""
import json

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
for name, stem in (("ER", "results_v6g10xm"), ("ER-conv", "results_v6g10n")):
    cells = []
    for B in ("", "B"):
        allv = []
        for F in (1, 2, 3, 4):
            per = {}
            for s in (1, 2, 3):
                d = json.load(open(f"{H}/{stem}{B}{'' if F == 1 else f'_f{F}'}_s{s}_dr0/eval_student_ZR.json"))
                for c in d["clips"]:
                    if c["split"] == "test":
                        per.setdefault(c["clip"], []).append(float(c["success_v2"]))
            v = [np.mean(x) for x in per.values()]
            allv += v
            cells.append(f"{100 * np.mean(v):.1f}")
        cells.append(f"{100 * np.mean(allv):.1f} (n={len(allv)})")
    print(name, " & ".join(cells))
