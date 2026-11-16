"""For every CrouchedTraverse reference: does the pelvis pass UNDER the beam (within its length), when, how deep is the
crouch there (highest body point vs beam underside), and how long the deep-crouch phase lasts."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
D = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"
split = json.load(open(f"{H}/data/clip_split_trial/split.json"))
where = {n: k for k in ("train", "val", "test") for n in split[k]}
man = json.load(open(f"{H}/runs/latent_v5/seed42/Z0/split_manifest.json"))
c2s = {f"{a}_{int(t):03d}": s for a, g in man["subject_trial_groups"].items() for s, ts in g.items() for t in ts}
print("trial            split subj      under_s(t)   n_under  top_min  under_z  pelvis_min  start_side end_side  len_s")
for f in sorted(glob.glob(f"{D}/*_motion.npz")):
    c = os.path.basename(f).replace("_motion.npz", "")
    m = np.load(os.path.realpath(f))
    p = json.load(open(f"{D}/{c}_bridge.json"))["parts"]["plank"]
    ctr, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    B = m["body_pos_w"]
    rel = B[:, :, :2] - ctr[:2]
    along, across = rel @ d, rel @ n
    # frames where any body point is within the beam footprint (across +-0.25 m, along within the plank)
    inside = (np.abs(across) < 0.25) & (np.abs(along) < half[0])
    fr = np.where(inside.any(1))[0]
    under_z = ctr[2] - half[2]
    top = np.where(inside, B[:, :, 2], -1).max(1)
    pel = B[:, 0]
    s0, s1 = np.sign((pel[0, :2] - ctr[:2]) @ n), np.sign((pel[-1, :2] - ctr[:2]) @ n)
    print(f"{c:16s} {where.get(c, '?'):5s} {c2s.get(c, '?'):9s} "
          f"{(f'{fr[0] / 50:.2f}-{fr[-1] / 50:.2f}' if len(fr) else '-'):12s} {len(fr):4d}     "
          f"{(top[fr].max() if len(fr) else np.nan):.2f}     {under_z:.2f}     {pel[:, 2].min():.2f}       "
          f"{s0:+.0f}         {s1:+.0f}        {len(pel) / 50:.2f}")
