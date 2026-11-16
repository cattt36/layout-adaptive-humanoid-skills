"""Signed landing error at each stake (along the step direction / lateral), from eval_expert --save-rollouts output.

  python stake_landing_error.py <rollout_dir> <stake_json_dir>
For every env and stake: the foot position at the first frame of the reference stance window where the foot is low
(ankle < top + 0.10) is compared with the stake centre; along = projection on the previous->this stake direction.
"""
import glob
import json
import os
import sys

import numpy as np

rd, sd = sys.argv[1:3]
rows = {}
for f in sorted(glob.glob(os.path.join(rd, "rollout_*.npz"))):
    clip = os.path.basename(f)[8:-4]
    j = os.path.join(sd, f"{clip}_stakes.json")
    if not os.path.exists(j) or clip.endswith("_push"):
        continue
    d = np.load(f)
    meta = json.load(open(j))
    fp, valid = d["feet_pos"], d["valid"]  # [n, E, 2, 3]
    top = float(os.environ.get("HNU_LAND_TOP", meta["height"]))
    prev = None
    land_prev = {}
    for i, s in enumerate(meta["stakes"]):
        k = 0 if s["foot"] == "L" else 1
        a = max(int(round(s["stance"][0] * 50 / meta["fps"])) - 1 - 8, 0)
        e = min(int(round(s["stance"][1] * 50 / meta["fps"])) - 1 + 8, len(fp) - 1)
        c = np.array([s["x"], s["y"]])
        dirv = (c - prev) / max(np.linalg.norm(c - prev), 1e-6) if prev is not None else None
        prev_c = prev
        prev = c
        for env in range(fp.shape[1]):
            seg = fp[a:e + 1, env, k]
            v = valid[a:e + 1, env]
            low = np.where(v & (seg[:, 2] < top + 0.10))[0]
            if not len(low):
                continue
            land = seg[low[0], :2]
            p = land - c
            if env in land_prev and dirv is not None:  # achieved / reference step length between landings
                ratio = np.linalg.norm(land - land_prev[env]) / np.linalg.norm(c - prev_c)
                rows.setdefault("ratio", []).append(ratio)
            land_prev[env] = land
            if dirv is None:
                rows.setdefault(i, []).append((np.nan, np.nan, np.linalg.norm(p)))
            else:
                along = p @ dirv
                lat = p @ np.array([-dirv[1], dirv[0]])
                rows.setdefault(i, []).append((along, lat, np.linalg.norm(p)))
ratio = np.array(rows.pop("ratio", [np.nan]))
print(f"achieved / reference step length between consecutive landings: median {np.nanmedian(ratio):.3f}, "
      f"mean {np.nanmean(ratio):.3f}, IQR [{np.nanpercentile(ratio, 25):.3f}, {np.nanpercentile(ratio, 75):.3f}], n {len(ratio)}")
for i, r in sorted(rows.items()):
    r = np.array(r)
    print(f"stake {i}: n {len(r)}  |err| median {np.median(r[:, 2]):.3f} m  along-track mean {np.nanmean(r[:, 0]):+.3f} m "
          f"(neg = short)  lateral mean {np.nanmean(r[:, 1]):+.3f} m  hit<0.15 {np.mean(r[:, 2] < 0.15):.2f}")
