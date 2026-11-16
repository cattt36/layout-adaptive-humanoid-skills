"""Summarise stakes evaluation jsons: success, all-stakes success, stake-hit rate, mean failure point."""
import glob
import json
import sys

import numpy as np

for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat, recursive=True)):
        j = json.load(open(f))
        fails = [x for c in j["clips"] for x in c["fail_frac_of_clip"]]
        g = lambda k: float("nan") if j.get(k) is None else j[k]
        print(f"{f}\n  success {g('success_rate'):.3f}  all_stakes {g('success_all_stakes'):.3f}  "
              f"hit {g('stake_hit_rate'):.3f}  anchor_xy {g('mean_anchor_xy_err'):.3f}  "
              f"n_fail {len(fails)}  median fail@ {np.median(fails) if fails else float('nan'):.2f}")
