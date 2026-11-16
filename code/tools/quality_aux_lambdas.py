"""Calibration-split AUROC and test AUROC / risk@80% per level and lambda (mean over seeds)."""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
for P in ["Z1", "Z2", "Z3", "Z4"]:
    for lam in ["0", "0.1", "0.3", "1.0"]:
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(R, P, f"lam{lam}_s*", "quality_eval.json"))]
        if rs:
            print(f"{P} λ={lam:4s} cal AUROC {np.mean([r['cal_auroc'] for r in rs]):.4f}  "
                  f"test AUROC {np.mean([r['isotonic']['auroc'] for r in rs]):.4f}  "
                  f"test risk@80 {100 * np.mean([r['isotonic']['risk@cov80'] for r in rs]):.2f}%")
