"""Validation future-prediction loss (and human_ctrl) of the v5 latents at the selected (best) epoch, per level and seed."""
import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1]
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    vals = []
    for f in sorted(glob.glob(os.path.join(root, "seed*", P, "history.json"))):
        h = json.load(open(f))
        best = min(h, key=lambda r: r.get("selection_score", 1e9))
        v = best.get("val", {})
        vals.append((v.get("future", np.nan), v.get("human_ctrl", np.nan), v.get("recon", np.nan), best["epoch"]))
    v = np.array(vals, float)
    print(f"{P}: val future {np.nanmean(v[:, 0]):.4f} ± {np.nanstd(v[:, 0]):.4f}   val recon {np.nanmean(v[:, 2]):.4f}   "
          f"best epochs {v[:, 3].astype(int).tolist()}")
