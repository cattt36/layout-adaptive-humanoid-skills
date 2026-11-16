"""EXPLORATORY (not pre-registered): linear probes of z_skill over the 4 CV folds x 3 latent seeds (extends
tools/probe_z.py, which used fold 1 / seed 42 only).  Same linear models for every level; train on the fold's
train-trial frames, evaluate on its test-trial frames.
  control  : next-stake position in the pelvis yaw frame (ThreeStepPoleRun / FiveStepPoleRun), time to the next touchdown, support state,
             pelvis displacement 0.5 s ahead (yaw frame), pelvis height
  semantic : action class
Input: z(t), z(t+0.5 s), z(t+1 s) concatenated (the student sees offsets up to 1 s).
  python probe_z_cv.py            -> runs/probe_z_cv.json + table (mean ± sd over the 12 encoders per level)"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H = "/root/gpufree-data/hnu_v2/data"
OUT = os.environ.get("HNU_PROBE_OUT", "/root/gpufree-data/hnu_v2/runs/probe_z_cv.json")
DIRS = {"ThreeStepPoleRun": f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/ref_bridge_video_lift/CrouchedTraverse"}
TASKS = list(DIRS)
LV = os.environ.get("HNU_PROBE_LEVELS", "Z0,Z1,Z2,Z3,Z4").split(",")  # "Z3K" = L3-kin (tools/run_l3kin.sh)
STEP = 2
KEYS_R = ("stake", "ttd", "disp", "height")
KEYS_C = ("support", "task")


def yaw_of(q):
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
    rise = np.zeros(T, bool)
    rise[1:] = (con[1:] & ~con[:-1]).any(1)
    nxt, last = np.full(T, np.inf), np.inf
    for i in range(T - 1, -1, -1):
        if rise[i]:
            last = i
        nxt[i] = last - i
    fut = np.minimum(np.arange(T) + 25, T - 1)
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
    return dict(stake=stake, ttd=np.minimum(nxt / 50.0, 1.0), support=con[:, 0].astype(int) + 2 * con[:, 1].astype(int),
                disp=to_yaw(pel[fut, :2] - pel[:, :2], yaw), height=pel[:, 2], task=np.full(T, TASKS.index(t)))


def zdir(F, S, P):
    if P == "Z3K":  # L3-kin ablation: z tables of the latents trained without human dynamics
        return f"{H}/z_tables_kin_f{F}_s{S}/Z3"
    if P == "Z3R":  # L3-rom ablation: without human dynamics and without human gait timing
        return f"{H}/z_tables_rom_f{F}_s{S}/Z3"
    if P == "Z3Q":  # factorisation ablation: L3 without executability supervision of z_quality
        return f"{H}/z_tables_noq_f{F}_s{S}/Z3"
    return f"{H}/z_tables_cv_f{F}/{P}" if S == 42 else f"{H}/z_tables_cv_f{F}_s{S}/{P}"


TG = {}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for F in (1, 2, 3, 4):
    sp = json.load(open(f"{H}/clip_split_cv/fold{F}/split.json"))
    tr, te = sorted(sp["train"]), sorted(sp["test"])
    for c in tr + te:
        if c not in TG:
            TG[c] = targets(c)
    for S in (42, 43, 44):
        for P in LV:
            key = f"f{F}_s{S}_{P}"
            if key in res or not os.path.isdir(zdir(F, S, P)):
                continue
            Z = {c: np.load(f"{zdir(F, S, P)}/{c}_z.npy") for c in tr + te}

            def feats(c):
                z = Z[c]
                T = len(z)
                return np.concatenate([z[np.minimum(np.arange(T) + o, T - 1)] for o in (0, 25, 50)], 1)[::STEP]
            Xtr, Xte = np.concatenate([feats(c) for c in tr]), np.concatenate([feats(c) for c in te])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            out = {}
            for k in KEYS_R:
                ytr = np.concatenate([TG[c][k][::STEP] for c in tr]).reshape(len(Xtr), -1)
                yte = np.concatenate([TG[c][k][::STEP] for c in te]).reshape(len(Xte), -1)
                mtr, mte = ~np.isnan(ytr).any(1), ~np.isnan(yte).any(1)
                out[k] = float(r2_score(yte[mte], Ridge(alpha=1.0).fit(Xtr[mtr], ytr[mtr]).predict(Xte[mte])))
            for k in KEYS_C:
                ytr = np.concatenate([TG[c][k][::STEP] for c in tr])
                yte = np.concatenate([TG[c][k][::STEP] for c in te])
                out[k] = float(balanced_accuracy_score(yte, LogisticRegression(max_iter=3000).fit(Xtr, ytr).predict(Xte)))
            res[key] = out
            json.dump(res, open(OUT, "w"), indent=1)
            print(key, {k: round(v, 3) for k, v in out.items()}, flush=True)
print("\n| level | n | next stake xy R2 | time to touchdown R2 | pelvis disp 0.5 s R2 | pelvis height R2 | support BA || action BA |")
print("|---|---|---|---|---|---|---|---|")
for P in LV:
    rows = [v for k, v in res.items() if k.endswith(P)]
    if not rows:
        continue
    f = lambda k: f"{np.mean([r[k] for r in rows]):.3f} ± {np.std([r[k] for r in rows], ddof=1) if len(rows) > 1 else 0:.3f}"  # noqa: E731
    print(f"| L{P[1]} | {len(rows)} | {f('stake')} | {f('ttd')} | {f('disp')} | {f('height')} | {f('support')} || {f('task')} |")
for k in KEYS_R + KEYS_C:
    d = [res[f"f{F}_s{S}_Z4"][k] - res[f"f{F}_s{S}_Z0"][k] for F in (1, 2, 3, 4) for S in (42, 43, 44)
         if f"f{F}_s{S}_Z4" in res and f"f{F}_s{S}_Z0" in res]
    if d:
        print(f"L4 - L0 {k}: mean {np.mean(d):+.3f}, L4 higher in {int(np.sum(np.array(d) > 0))}/{len(d)}")
