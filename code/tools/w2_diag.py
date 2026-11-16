"""Why is L4 weak on some FiveStepPoleRun val / test trials?  Per trial: subject, length, approach speed, stake geometry
(gaps, lateral offsets), stance durations, and how far each level's z sequence lies from the TRAINING z set
(per frame nearest-neighbour distance to all train-trial frames, normalised by the same score of the train trials
in a leave-one-trial-out sense)."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
D = f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun"
split = json.load(open(f"{H}/data/clip_split_trial/split.json"))
where = {n: k for k in ("train", "val", "test") for n in split[k]}
man = json.load(open(f"{H}/runs/latent_v5/seed42/Z0/split_manifest.json"))
c2s = {f"{a}_{int(t):03d}": s for a, g in man["subject_trial_groups"].items() for s, ts in g.items() for t in ts}
clips = sorted(os.path.basename(f).replace("_motion.npz", "") for f in glob.glob(f"{D}/*_motion.npz"))

print("trial            split subj      len_s  v_app  gaps_m (along path)              lat_m  stance_s  stride_s")
geo = {}
for c in clips:
    m = np.load(f"{D}/{c}_motion.npz")
    j = json.load(open(f"{D}/{c}_stakes.json"))
    fps = float(m["fps"])
    T = m["joint_pos"].shape[0]
    st = j["stakes"]
    xy = np.array([[s["x"], s["y"]] for s in st])
    gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # lateral spread: distance of the stakes from the line first -> last stake
    d = xy[-1] - xy[0]
    d = d / np.linalg.norm(d)
    lat = np.abs((xy - xy[0]) @ np.array([-d[1], d[0]]))
    sf = float(j["fps"])
    stance = np.mean([(s["stance"][1] - s["stance"][0]) / sf for s in st])
    stride = np.mean(np.diff([s["stance"][0] for s in st])) / sf
    f0 = int(st[0]["stance"][0] / sf * fps)
    v = np.linalg.norm(m["body_lin_vel_w"][max(0, f0 - 25):f0, 0, :2], axis=-1).mean()
    geo[c] = dict(gaps=gaps, lat=lat.max(), stance=stance, stride=stride, v=v, T=T / fps)
    print(f"{c:16s} {where.get(c, '?'):5s} {c2s.get(c, '?'):9s} {T / fps:5.1f}  {v:4.2f}   "
          f"{' '.join(f'{g:.2f}' for g in gaps):30s} {lat.max():5.2f}  {stance:5.2f}     {stride:5.2f}")

tr = [c for c in clips if where.get(c) == "train"]
for k in ("gaps", "lat", "stance", "stride", "v"):
    a = np.array([np.mean(geo[c][k]) for c in tr])
    print(f"train {k:6s} mean {a.mean():.3f} sd {a.std():.3f}")

print("\nz novelty (median per-frame NN distance to train frames / train leave-one-out median); >1.5 = unusual")
lv = ["Z0", "Z1", "Z2", "Z3", "Z4"]
print("trial            split " + " ".join(f"{p:>6s}" for p in lv))
res = {}
for P in lv:
    Z = {c: np.load(f"{H}/data/z_tables_v5t/{P}/{c}_z.npy").reshape(-1, np.load(f"{H}/data/z_tables_v5t/{P}/{c}_z.npy").shape[-1])
         for c in clips if os.path.exists(f"{H}/data/z_tables_v5t/{P}/{c}_z.npy")}
    sub = {c: z[::4] for c, z in Z.items()}  # 4x subsampled frames

    def score(c, pool):
        P_ = np.concatenate([sub[x] for x in pool])
        d = np.sqrt(((sub[c][:, None] - P_[None]) ** 2).sum(-1)).min(1)
        return np.median(d)
    loo = np.median([score(c, [x for x in tr if x != c]) for c in tr])
    for c in clips:
        if c in sub:
            res.setdefault(c, {})[P] = score(c, [x for x in tr if x != c]) / loo
for c in clips:
    print(f"{c:16s} {where.get(c, '?'):5s} " + " ".join(f"{res[c].get(P, np.nan):6.2f}" for P in lv))
