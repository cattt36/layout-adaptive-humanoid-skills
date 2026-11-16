"""Check of the v6g10k training set: every allowed clip has a motion file and, per level, a z table of equal length;
1.0x re-export reproduces z_tables_v5t; encoder z of the 0.9x copies vs interpolated 1.0x z (information only)."""
import glob
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
SFX = sys.argv[1] if len(sys.argv) > 1 else ""  # "" (v6g10k) or "_m" (v6g10m)
allow = [ln.strip() for ln in open(f"{H}/clip_split_trial/clips_train_aug{SFX}.txt") if ln.strip()]
dirs = {"ThreeStepPoleRun": f"{H}/aug_ts{SFX}/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/aug_ts{SFX}/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/aug_ts{SFX}/ref_bridge_video_lift/CrouchedTraverse" if SFX else f"{H}/ref_bridge_video_lift/CrouchedTraverse"}
bad = 0
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    for c in allow:
        m = f"{dirs[c.split('_')[0]]}/{c}_motion.npz"
        z = f"{H}/z_tables_v5t_aug{SFX}/{P}/{c}_z.npy"
        if not (os.path.exists(m) and os.path.exists(z)):
            print("missing", P, c, os.path.exists(m), os.path.exists(z))
            bad += 1
            continue
        T, Tz = np.load(m)["joint_pos"].shape[0], np.load(z).shape[0]
        if T != Tz:
            print("length", P, c, T, Tz)
            bad += 1
    d = [np.abs(np.load(f) - np.load(f"{H}/z_tables_v5t/{P}/{os.path.basename(f)}")).max()
         for f in glob.glob(f"{H}/ts/ts100/z_enc/{P}/*_z.npy")]
    r = []
    for f in glob.glob(f"{H}/ts/ts090/z_enc/{P}/*_z.npy"):
        g = f"{H}/ts/ts090/z_tables_v5t/{P}/{os.path.basename(f)}"
        if os.path.exists(g):
            a, b = np.load(f), np.load(g)
            r.append(np.linalg.norm(a - b, axis=-1).mean() / np.linalg.norm(b, axis=-1).mean())
    print(f"{P}: 1.0x reproduction max diff {max(d):.1e}; 0.9x encoder z vs interpolated z: relative diff {np.mean(r):.3f}")
print(f"{len(allow)} training clips, problems: {bad}")
