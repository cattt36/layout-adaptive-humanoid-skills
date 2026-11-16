"""Reference ankle-link height (body 'left/right_ankle_roll') when standing on a stake (stance windows) vs on the
ground (contact label, outside stake windows), and the swing height over stakes -> on-stake height band."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
LA, RA = 6, 12  # left / right ankle_roll in the 30-body list
on_st, on_gr, swing_over = [], [], []
for d in (f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun", f"{H}/ref_stakes_s83_h10/FiveStepPoleRun"):
    for f in sorted(glob.glob(f"{d}/*_motion.npz")):
        m = np.load(os.path.realpath(f))
        j = json.load(open(f.replace("_motion.npz", "_stakes.json")))
        p = m["body_pos_w"]
        con = m["foot_contact"]
        mask = np.zeros(len(p), bool)
        for s in j["stakes"]:
            a, b = int(s["stance"][0] * 50 / j["fps"]), int(s["stance"][1] * 50 / j["fps"])
            k = LA if s["foot"] == "L" else RA
            on_st += list(p[a:b, k, 2])
            mask[a:b] = True
            other = RA if k == LA else LA  # the other foot near this stake while in the air
            xy = np.array([s["x"], s["y"]])
            dd = np.linalg.norm(p[:, other, :2] - xy, axis=-1)
            swing_over += list(p[(dd < 0.17) & ~con[:, 1 if other == RA else 0], other, 2])
        for k, ci in ((LA, 0), (RA, 1)):
            on_gr += list(p[~mask & con[:, ci], k, 2])
q = lambda v: np.percentile(v, [5, 50, 95]).round(3) if len(v) else None  # noqa: E731
print("ankle z on stake  (5/50/95 %):", q(on_st))
print("ankle z on ground (5/50/95 %):", q(on_gr))
print("other foot in the air within 0.17 m of a stake centre (5/50/95 %):", q(swing_over), "n", len(swing_over))
