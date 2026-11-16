"""Data-quality screen of every trial, from the DATA ONLY (no model results):
  ThreeStepPoleRun / FiveStepPoleRun  stake gaps (distance between consecutive stakes) vs the action's median gap; number of stakes;
           reference foot height on each stake during its stance (foot must be on the stake top)
  DuMu     bridge registration error (fit_err_m), beam underside height, the pelvis must pass under the beam
  all      clip length, reference pelvis height range
Flags are printed with their reason; thresholds are fixed here before looking at any per-trial results list."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
split = json.load(open(f"{H}/clip_split_trial/split.json"))
where = {n: k for k in ("train", "val", "test") for n in split[k]}
LA, RA = 6, 12
flags = {}
for t, d in (("ThreeStepPoleRun", f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun"), ("FiveStepPoleRun", f"{H}/ref_stakes_s83_h10/FiveStepPoleRun")):
    rows = []
    for f in sorted(glob.glob(f"{d}/*_stakes.json")):
        c = os.path.basename(f).replace("_stakes.json", "")
        j = json.load(open(f))
        m = np.load(os.path.realpath(f.replace("_stakes.json", "_motion.npz")))
        xy = np.array([[s["x"], s["y"]] for s in j["stakes"]])
        gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        fz = []
        for s in j["stakes"]:
            a, b = int(s["stance"][0] * 50 / j["fps"]), int(s["stance"][1] * 50 / j["fps"])
            k = LA if s["foot"] == "L" else RA
            fz.append(float(np.median(m["body_pos_w"][a:b + 1, k, 2])) if b > a else np.nan)
        rows.append((c, gaps, len(j["stakes"]), fz))
    med = np.median(np.concatenate([g for _, g, _, _ in rows]))
    print(f"== {t}: median stake gap {med:.2f} m")
    for c, gaps, n, fz in rows:
        why = []
        if (gaps > 1.5 * med).any() or (gaps < 0.5 * med).any():
            why.append(f"stake gap {' '.join(f'{g:.2f}' for g in gaps)} m (median {med:.2f}; allowed 0.5-1.5x)")
        if any(z < 0.12 for z in fz if z == z):
            why.append(f"reference foot not on a stake top during its stance (ankle z {' '.join(f'{z:.2f}' for z in fz)})")
        if why:
            flags[c] = why
        print(f"   {c} [{where.get(c, '?')}] gaps {' '.join(f'{g:.2f}' for g in gaps)}  ankle-on-stake z {' '.join(f'{z:.2f}' for z in fz)}"
              + ("   <-- FLAG" if why else ""))
print("== CrouchedTraverse")
D = f"{H}/ref_bridge_video_lift/CrouchedTraverse"
for f in sorted(glob.glob(f"{D}/*_bridge.json")):
    c = os.path.basename(f).replace("_bridge.json", "")
    b = json.load(open(os.path.realpath(f)))
    m = np.load(os.path.realpath(f.replace("_bridge.json", "_motion.npz")))
    p = b["parts"]["plank"]
    ctr, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    dd, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - ctr[:2]
    inside = (np.abs(rel @ n) < 0.25) & (np.abs(rel @ dd) < half[0])
    under = ctr[2] - half[2]
    top_inside = np.where(inside, m["body_pos_w"][:, :, 2], -1).max()
    why = []
    if b.get("fit_err_m", 0) > 0.05:
        why.append(f"bridge registration error {b['fit_err_m']:.3f} m > 0.05")
    if top_inside > under + 0.02:
        why.append(f"a body point is {top_inside - under:.2f} m above the beam underside while under the beam (penetration)")
    if not inside.any():
        why.append("never passes under the beam")
    if why:
        flags[c] = why
    print(f"   {c} [{where.get(c, '?')}] fit_err {b.get('fit_err_m', float('nan')):.3f}  beam underside {under:.2f}  "
          f"highest body point under the beam {top_inside:.2f}" + ("   <-- FLAG" if why else ""))
print("\nFLAGGED:")
for c, w in flags.items():
    print(f"   {c} [{where.get(c, '?')}]: " + "; ".join(w))
