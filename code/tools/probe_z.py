"""EXPLORATORY (not pre-registered): linear probes of control-relevant vs semantic quantities from z_skill.
Question (reviewer): latent quality improves L0 -> L4, why not the closed-loop success?  Hypothesis: semantic
supervision changes how the latent is ORGANISED (semantics linearly readable) more than how much CONTROL information it
carries.  Same linear models for every level; train on train-trial frames, evaluate on test-trial frames.
  control  : next-stake position in the pelvis yaw frame (ThreeStepPoleRun / FiveStepPoleRun), time to the next touchdown, support state
             (none / left / right / both), pelvis displacement 0.5 s ahead (yaw frame), pelvis height
  semantic : action class
Inputs: z(t) alone, and z(t), z(t+0.5 s), z(t+1 s) concatenated (the student sees 9 offsets up to 1 s)."""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

H = "/root/gpufree-data/hnu_v2/data"
DIRS = {"ThreeStepPoleRun": f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/ref_bridge_video_lift/CrouchedTraverse"}
TASKS = list(DIRS)
split = json.load(open(f"{H}/clip_split_trial/split.json"))
where = {n: k for k in ("train", "val", "test") for n in split[k]}
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
STEP = 2


def yaw_of(q):  # wxyz
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def to_yaw(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], -1)


def targets(clip):
    t = clip.split("_")[0]
    m = np.load(os.path.realpath(f"{DIRS[t]}/{clip}_motion.npz"))
    T = m["joint_pos"].shape[0]
    pel = m["body_pos_w"][:, 0]
    yaw = yaw_of(m["body_quat_w"][:, 0])
    con = m["foot_contact"].astype(bool)
    # time to next touchdown (rising edge of either foot), clipped at 1 s
    rise = np.zeros(T, bool)
    rise[1:] = (con[1:] & ~con[:-1]).any(1)
    nxt = np.full(T, np.inf)
    last = np.inf
    for i in range(T - 1, -1, -1):
        if rise[i]:
            last = i
        nxt[i] = last - i
    ttd = np.minimum(nxt / 50.0, 1.0)
    support = con[:, 0].astype(int) + 2 * con[:, 1].astype(int)  # 0 none, 1 L, 2 R, 3 both
    fut = np.minimum(np.arange(T) + 25, T - 1)
    disp = to_yaw(pel[fut, :2] - pel[:, :2], yaw)
    stake = np.full((T, 2), np.nan)
    sj = f"{DIRS[t]}/{clip}_stakes.json"
    if os.path.exists(sj):
        j = json.load(open(sj))
        ends = [s["stance"][1] * 50 / j["fps"] for s in j["stakes"]]
        xy = np.array([[s["x"], s["y"]] for s in j["stakes"]])
        for i in range(T):
            k = next((n for n, e in enumerate(ends) if e >= i), None)
            if k is not None:
                stake[i] = to_yaw((xy[k] - pel[i, :2])[None], yaw[i:i + 1])[0]
    return dict(stake=stake, ttd=ttd, support=support, disp=disp, height=pel[:, 2], task=np.full(T, TASKS.index(t)))


clips = sorted(c for c in where if c.split("_")[0] in TASKS and where[c] in ("train", "test"))
TG = {c: targets(c) for c in clips}
res = {}
for P in LV:
    Z = {c: np.load(f"{H}/z_tables_v5t/{P}/{c}_z.npy") for c in clips}
    for variant in ("z(t)", "z(t,+0.5s,+1s)"):
        def feats(c):
            z = Z[c]
            if variant == "z(t)":
                return z
            T = len(z)
            return np.concatenate([z[np.minimum(np.arange(T) + o, T - 1)] for o in (0, 25, 50)], 1)
        X = {c: feats(c)[::STEP] for c in clips}
        Y = {c: {k: v[::STEP] for k, v in TG[c].items()} for c in clips}
        tr = [c for c in clips if where[c] == "train"]
        te = [c for c in clips if where[c] == "test"]
        Xtr, Xte = np.concatenate([X[c] for c in tr]), np.concatenate([X[c] for c in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        out = {}
        for key in ("stake", "ttd", "disp", "height"):
            ytr = np.concatenate([Y[c][key] for c in tr]).reshape(len(Xtr), -1)
            yte = np.concatenate([Y[c][key] for c in te]).reshape(len(Xte), -1)
            mtr, mte = ~np.isnan(ytr).any(1), ~np.isnan(yte).any(1)
            r = Ridge(alpha=1.0).fit(Xtr[mtr], ytr[mtr])
            out[key] = r2_score(yte[mte], r.predict(Xte[mte]))
        for key in ("support", "task"):
            ytr = np.concatenate([Y[c][key] for c in tr])
            yte = np.concatenate([Y[c][key] for c in te])
            clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
            out[key] = balanced_accuracy_score(yte, clf.predict(Xte))
        res[(P, variant)] = out
for variant in ("z(t)", "z(t,+0.5s,+1s)"):
    print(f"\n== input {variant}; test trials.  R2 (regression) / balanced accuracy (classification)")
    print("level | next stake xy | time to touchdown | pelvis disp 0.5s | pelvis height | support state || action class")
    for P in LV:
        o = res[(P, variant)]
        print(f"L{P[1]}    |     {o['stake']:.3f}     |       {o['ttd']:.3f}       |      {o['disp']:.3f}       |     "
              f"{o['height']:.3f}     |     {o['support']:.3f}     ||    {o['task']:.3f}")
