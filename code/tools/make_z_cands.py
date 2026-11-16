"""Experiment A (full chain, runs/preregistration_expA_impl.md): candidate-sweep z tables.  Table k gives every clip
of action a the z_skill of the k-th LIBRARY clip of a (fold training clips, sorted), linearly resampled to the clip's
length; clips whose action library has fewer than k+1 clips, the clip itself as candidate, and clips without
simulation geometry keep their own z (such pairs are not used).  Writes data/z_cand_f<k>/k<kk>/Z*/ and cands.json.
  python make_z_cands.py <fold>"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SRC = f"{H}/data/z_tables_cv_f{F}"
DST = f"{H}/data/z_cand_f{F}"
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
split = json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))
allnames = sorted(f[:-6] for f in os.listdir(os.path.join(SRC, "Z0")) if f.endswith("_z.npy"))
geo = {n for n in allnames if os.path.exists(os.path.join(GEO[n.split("_")[0]], f"{n}_motion.npz"))}
LIB = {t: sorted(n for n in split["train"] if n.startswith(t)) for t in GEO}
K = max(len(v) for v in LIB.values())


def resample(z, T):
    src = np.linspace(0.0, 1.0, len(z))
    dst = np.linspace(0.0, 1.0, T)
    return np.stack([np.interp(dst, src, z[:, j]) for j in range(z.shape[1])], 1).astype(z.dtype)


os.makedirs(DST, exist_ok=True)
json.dump(dict(library=LIB, K=K, geometry_clips=sorted(geo)), open(os.path.join(DST, "cands.json"), "w"), indent=1)
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    d = os.path.join(SRC, P)
    Z = {n: np.load(os.path.join(d, f"{n}_z.npy")) for n in allnames}
    for k in range(K):
        o = os.path.join(DST, f"k{k:02d}", P)
        os.makedirs(o, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith("_z.npy") and os.path.isfile(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), o)
        for n in allnames:
            lib = LIB[n.split("_")[0]]
            c = lib[k] if (n in geo and k < len(lib)) else n
            out = Z[n] if c == n else resample(Z[c], len(Z[n]))
            np.save(os.path.join(o, f"{n}_z.npy"), out)
print({t: len(v) for t, v in LIB.items()}, "K =", K)
