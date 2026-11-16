"""Q_exec (executability predictor, EXPLORATORY item 5) of the fold-1 seed-1 students: test AUROC / risk at 80 %
coverage (isotonic), label-scarce (5 % of the training envs) and full data, 3 training seeds each."""
import glob
import json
import sys

import numpy as np

R = "/root/gpufree-data/hnu_v2/runs/quality_v6g10n_s" + (sys.argv[1] if len(sys.argv) > 1 else "1")
print("| level | data | AUROC | risk @ 80 % coverage | no-filter failure rate | heuristic AUROC |")
print("|---|---|---|---|---|---|")
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    for fr in ("0.05", "1.0"):
        r = [json.load(open(f)) for f in glob.glob(f"{R}/{P}/frac{fr}_s*/quality_eval.json")]
        if r:
            a = [x["isotonic"]["auroc"] for x in r]
            k = [100 * x["isotonic"]["risk@cov80"] for x in r]
            print(f"| L{P[1]} | {'5 %' if fr == '0.05' else '100 %'} | {np.mean(a):.3f} ± {np.std(a):.3f} | {np.mean(k):.1f}% | "
                  f"{100 * r[0]['no_filter_risk']:.1f}% | {r[0]['heuristic']['auroc']:.3f} |")
