"""Q_exec with level-allowed auxiliary supervision: lambda chosen per level on the calibration split; test metrics
(isotonic-calibrated) for lambda = 0 (no auxiliary) and the chosen lambda, mean +- std over 3 training seeds."""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
print("| Level | aux targets (human) | chosen λ (cal) | AUROC λ=0 | AUROC chosen | AUPRC(fail) λ=0 | AUPRC chosen | "
      "Risk@80% λ=0 | Risk@80% chosen | ECE chosen |")
print("|---|---|---|---|---|---|---|---|---|---|")
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    runs = {}
    for f in glob.glob(os.path.join(R, P, "lam*_s*", "quality_eval.json")):
        lam = os.path.basename(os.path.dirname(f)).split("_")[0][3:]
        runs.setdefault(lam, []).append(json.load(open(f)))
    if not runs:
        continue
    best = max(runs, key=lambda k: np.mean([r["cal_auroc"] for r in runs[k]]))
    def ms(lam, key, sub="isotonic", scale=1.0):  # noqa: E301
        v = [r[sub][key] * scale for r in runs[lam]]
        return f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    r0 = runs["0"][0]
    print(f"| L{P[1]} | {r0.get('n_aux_targets', 0) if best == '0' else runs[best][0]['n_aux_targets']} "
          f"({runs[best][0]['n_human_aux_targets']}) | {best} | {ms('0', 'auroc')} | {ms(best, 'auroc')} | "
          f"{ms('0', 'auprc_failure')} | {ms(best, 'auprc_failure')} | {ms('0', 'risk@cov80', scale=100)} | "
          f"{ms(best, 'risk@cov80', scale=100)} | {ms(best, 'ece')} |")
