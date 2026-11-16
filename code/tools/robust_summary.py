"""Item 3 (robustness): success per action and level, mean ± std over the RL seeds, for one condition.
  python tools/robust_summary.py <V> <dr 0|1>
"""
import json
import os
import sys

import numpy as np

V, DR = sys.argv[1], sys.argv[2]
H = "/root/gpufree-data/hnu_v2/runs"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
print(f"### {V}, {'randomised' if DR == '1' else 'nominal'} environment: success % (mean ± std over seeds)\n")
print("| level | " + " | ".join(f"{t} clean | {t} push" for t in TASKS) + " |\n|---|" + "---|---|" * len(TASKS))
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    cells = []
    for T in TASKS:
        for x in ("", "_push"):
            v = [json.load(open(f))["summary"][f"{T}/all"]["success"] for s in (1, 2, 3)
                 if os.path.exists(f := f"{H}/results_{V}_s{s}_dr{DR}/eval_student_{P}{x}.json")]
            cells.append(f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}" if v else "-")
    print(f"| L{P[1]} | " + " | ".join(cells) + " |")
