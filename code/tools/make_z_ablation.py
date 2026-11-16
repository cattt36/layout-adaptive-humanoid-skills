"""z-usage ablation (user request; diagnostic): altered z_skill tables for evaluating EXISTING students.
For every level Z0-Z4 of a fold's z tables (data/z_tables_cv_f<k>/Z*), writes data/z_abl_f<k>/<variant>/Z*:
  freeze : z[t] = z[0] of the clip for every frame (no temporal information)
  mean   : the mean z over all frames of the fold's TRAINING clips, for every frame of every clip (no clip information)
  swap   : z of another clip of the same action (next clip in name order, cyclic), linearly resampled in normalised
           time to the clip's length (plausible z with the right phase structure, but the wrong trial)
Other files of the table directory (split.json, ...) are copied unchanged.
  python make_z_ablation.py <fold>"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SRC = f"{H}/data/z_tables_cv_f{F}"
DST = f"{H}/data/z_abl_f{F}"
train = set(json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))["train"])


def resample(z, T):
    src = np.linspace(0.0, 1.0, len(z))
    dst = np.linspace(0.0, 1.0, T)
    return np.stack([np.interp(dst, src, z[:, j]) for j in range(z.shape[1])], 1).astype(z.dtype)


for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    d = os.path.join(SRC, P)
    names = sorted(f[:-6] for f in os.listdir(d) if f.endswith("_z.npy"))
    Z = {n: np.load(os.path.join(d, f"{n}_z.npy")) for n in names}
    mu = np.concatenate([Z[n] for n in names if n in train]).mean(0)
    by_task = {}
    for n in names:
        by_task.setdefault(n.split("_")[0], []).append(n)
    for v in ("freeze", "mean", "swap"):
        o = os.path.join(DST, v, P)
        os.makedirs(o, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith("_z.npy") and os.path.isfile(os.path.join(d, f)):  # (sub-directories such as _prep skipped)
                shutil.copy(os.path.join(d, f), o)
        for n in names:
            z = Z[n]
            if v == "freeze":
                out = np.repeat(z[:1], len(z), 0)
            elif v == "mean":
                out = np.repeat(mu[None].astype(z.dtype), len(z), 0)
            else:
                grp = by_task[n.split("_")[0]]
                out = resample(Z[grp[(grp.index(n) + 1) % len(grp)]], len(z))
            assert out.shape == z.shape
            np.save(os.path.join(o, f"{n}_z.npy"), out)
    print(P, len(names), "clips, D =", Z[names[0]].shape[1], "train clips in mean:", sum(n in train for n in names))
