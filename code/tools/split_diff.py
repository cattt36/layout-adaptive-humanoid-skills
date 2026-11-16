"""Why does nothing transfer to the test subjects?  Compare the robot reference clips of the train / val / test
splits per action: duration, pelvis speed and height, start state, joint ranges, stake / bridge layout, z tables."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
DIRS = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
sp = {}
for k in ("train", "val", "test"):
    for c in open(f"{H}/data/clip_split_v5/clips_{k}.txt").read().split():
        sp[c] = k
for act, d in DIRS.items():
    print(f"\n===== {act}")
    rows = {}
    for f in sorted(glob.glob(f"{d}/*_motion.npz")):
        c = os.path.basename(f).replace("_motion.npz", "")
        z = np.load(f)
        p = z["body_pos_w"][:, 0]
        v = np.linalg.norm(z["body_lin_vel_w"][:, 0, :2], axis=1)
        q = z["joint_pos"]
        stake = f.replace("_motion.npz", "_stakes.json")
        ns, gap = 0, np.nan
        if os.path.exists(stake):
            j = json.load(open(stake))
            xy = np.array(j["stake_xyz_50hz"])[:, :2]
            ns = len(xy)
            gap = np.mean(np.linalg.norm(np.diff(xy, axis=0), axis=1)) if ns > 1 else np.nan
        bridge = os.path.exists(f.replace("_motion.npz", "_bridge.json"))
        zt = f"{H}/data/z_tables_v5/Z0/{c}_z.npy"
        zn = np.linalg.norm(np.load(zt), axis=1).mean() if os.path.exists(zt) else np.nan
        rows.setdefault(sp.get(c, "?"), []).append(dict(
            T=len(p) / 50, speed_max=v.max(), speed_mean=v.mean(), h_mean=p[:, 2].mean(), h_min=p[:, 2].min(),
            h0=p[0, 2], v0=v[0], disp=np.linalg.norm(p[-1, :2] - p[0, :2]), knee_max=np.rad2deg(q[:, [3, 9]].max()),
            waist_pitch=np.rad2deg(q[:, 14].max()), stakes=ns, stake_gap=gap, bridge=float(bridge), znorm=zn))
    keys = list(next(iter(rows.values()))[0])
    print("split  n   " + "  ".join(f"{k:>10s}" for k in keys))
    for s in ("train", "val", "test", "?"):
        if s in rows:
            r = rows[s]
            print(f"{s:5s} {len(r):3d}  " + "  ".join(f"{np.nanmean([x[k] for x in r]):10.3f}" for k in keys))
