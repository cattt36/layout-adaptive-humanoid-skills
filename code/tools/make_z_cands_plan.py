"""Full chain with the new execution layer (user decision): candidate-sweep tables WITH the geometric
planner (rules of tools/make_z_plan.py plan4), replacing tools/make_z_cands.py (linear z + the trial's human clock).
Table k gives every clip of action a the k-th LIBRARY clip of a (fold training clips, sorted) as candidate:
  ThreeStepPoleRun / FiveStepPoleRun : planned stance windows from the candidate's swing / stance durations scaled by the stake distances, first
            onset from the per-action regression on the training clips; z = candidate z warped to the planned windows;
            the environment clock of that clip in table k = the planned windows (data/geom_cand_f<k>/k<kk>/...).
  beam    : entry under the beam predicted from the distance (regression on the training clips); z = candidate z warped
            so that its own entry frame falls on the prediction (no environment clock).
Pairs where the clip is its own candidate, the stake counts differ or the clip has no geometry keep the clip's own z
and clock and are marked invalid (never used).  Writes data/z_cand_plan_f<k>/k<kk>/Z0..Z3, geometry dirs, pairs.json.
  python make_z_cands_plan.py <fold>"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SRC = f"{H}/data/z_tables_cv_f{F}"
DST = f"{H}/data/z_cand_plan_f{F}"
GDST = f"{H}/data/geom_cand_f{F}"
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10"}
BRIDGE = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"
split = json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))
SPL = {n: k for k in ("train", "val", "test") for n in split[k]}
allnames = sorted(f[:-6] for f in os.listdir(os.path.join(SRC, "Z0")) if f.endswith("_z.npy"))


def has_geo(n):
    t = n.split("_")[0]
    return os.path.exists(os.path.join(BRIDGE if t == "CrouchedTraverse" else os.path.join(GEO[t], t), f"{n}_motion.npz"))


names = [n for n in allnames if has_geo(n)]
LIB = {t: sorted(n for n in split["train"] if n.startswith(t) and n in names) for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")}
K = max(len(v) for v in LIB.values())


def stakes(name):
    t = name.split("_")[0]
    j = json.load(open(os.path.join(GEO[t], t, f"{name}_stakes.json")))
    ref = np.load(os.path.join(GEO[t], t, f"{name}_motion.npz"))
    pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
    win = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
    return pts, ref["body_pos_w"][0, 0, :2], win, len(ref["body_pos_w"])


def beam(name):
    m = np.load(os.path.join(BRIDGE, f"{name}_motion.npz"))
    p = json.load(open(os.path.realpath(os.path.join(BRIDGE, f"{name}_bridge.json"))))["parts"]["plank"]
    c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, nn = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - c[:2]
    inside = (np.abs(rel @ nn) < 0.25) & (np.abs(rel @ d) < half[0])
    entry = int(np.where(inside.any(1))[0][0]) if inside.any() else -1
    p0 = m["body_pos_w"][0, 0, :2]
    return float(min(np.linalg.norm(c[:2] + s * d * half[0] - p0) for s in (1, -1))), entry, len(m["body_pos_w"])


ST = {n: stakes(n) for n in names if not n.startswith("CrouchedTraverse")}
BM = {n: beam(n) for n in names if n.startswith("CrouchedTraverse")}
REG = {t: np.polyfit([np.linalg.norm(ST[n][0][0] - ST[n][1]) for n in LIB[t]], [ST[n][2][0, 0] for n in LIB[t]], 1)
       for t in GEO}
bl = [n for n in LIB["CrouchedTraverse"] if BM[n][1] > 0]
RD = np.polyfit([BM[n][0] for n in bl], [BM[n][1] for n in bl], 1)


def plan_w(dst, src):
    pd, p0d, _, Td = ST[dst]
    ps, p0s, ws, _ = ST[src]
    seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
    r = np.clip(seg(p0d, pd) / np.maximum(seg(p0s, ps), 1e-3), 0.6, 1.6)
    gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]
    stance = ws[:, 1] - ws[:, 0]
    out, t = [], 0.0
    for k in range(len(ws)):
        a = float(np.polyval(REG[dst.split("_")[0]], np.linalg.norm(pd[0] - p0d))) if k == 0 else t + gaps[k] * r[k]
        out.append((a, a + stance[k]))
        t = a + stance[k]
    out = np.array(out)
    if out[-1, 1] > Td - 11:
        out *= (Td - 11) / out[-1, 1]
    return out, ws


def warp(z, T, ad, asrc):
    if not (np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)):
        ad, asrc = [0.0, T - 1.0], [0.0, len(z) - 1.0]
    tt = np.clip(np.interp(np.arange(T), ad, asrc), 0, len(z) - 1)
    return np.stack([np.interp(tt, np.arange(len(z)), z[:, j]) for j in range(z.shape[1])], 1).astype(z.dtype)


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


pairs, plans = {}, {}
for k in range(K):
    for n in names:
        t = n.split("_")[0]
        lib = LIB[t]
        if k >= len(lib) or lib[k] == n:
            continue
        c = lib[k]
        if t == "CrouchedTraverse":
            if BM[n][1] <= 0 or BM[c][1] <= 0:
                continue
            pred = float(np.clip(np.polyval(RD, BM[n][0]), 10, BM[n][2] - 30))
            plans[(k, n)] = ("beam", pred, float(BM[c][1]))
        else:
            if len(ST[n][0]) != len(ST[c][0]):
                continue
            pw, sw = plan_w(n, c)
            plans[(k, n)] = ("stakes", pw, sw)
        pairs.setdefault(f"k{k:02d}", {})[n] = c
os.makedirs(DST, exist_ok=True)
# HNU_SKIP_GEOM=1 (with HNU_LEVELS=ZR): only z tables of the given levels; pairs.json / geometry untouched
SKIP_GEOM = os.environ.get("HNU_SKIP_GEOM", "0") == "1"
if not SKIP_GEOM:
    json.dump(dict(library=LIB, K=K, split={n: SPL.get(n, "?") for n in names}, pairs=pairs), open(f"{DST}/pairs.json", "w"), indent=1)
# per-table geometry (environment clock of the stake clips = planned windows of that table's candidate)
for k in ([] if SKIP_GEOM else range(K)):
    for t, g in GEO.items():
        o = os.path.join(GDST, f"k{k:02d}", os.path.basename(g), t)
        os.makedirs(o, exist_ok=True)
        for f in os.listdir(os.path.join(g, t)):
            n = f[:-len("_stakes.json")] if f.endswith("_stakes.json") else None
            dst = os.path.join(o, f)
            if (k, n) in plans:
                j = json.load(open(os.path.join(g, t, f)))
                for s, (a, b) in zip(j["stakes"], plans[(k, n)][1]):
                    s["stance"] = [int(round(a * j["fps"] / 50.0)), int(round(b * j["fps"] / 50.0))]
                j["planned_from"] = LIB[t][k]
                if os.path.lexists(dst):
                    os.remove(dst)
                json.dump(j, open(dst, "w"))
            else:
                link(os.path.join(g, t, f), dst)
for P in os.environ.get("HNU_LEVELS", "Z0,Z1,Z2,Z3").split(","):
    d = os.path.join(SRC, P)
    Z = {n: np.load(os.path.join(d, f"{n}_z.npy")) for n in allnames}
    for k in range(K):
        o = os.path.join(DST, f"k{k:02d}", P)
        os.makedirs(o, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith("_z.npy") and os.path.isfile(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), o)
        for n in allnames:
            p = plans.get((k, n))
            if p is None:
                out = Z[n]
            elif p[0] == "beam":
                c = LIB["CrouchedTraverse"][k]
                out = warp(Z[c], len(Z[n]), [0.0, p[1], len(Z[n]) - 1.0], [0.0, p[2], len(Z[c]) - 1.0])
            else:
                c = LIB[n.split("_")[0]][k]
                out = warp(Z[c], len(Z[n]), [0.0] + p[1].flatten().tolist() + [len(Z[n]) - 1.0],
                           [0.0] + p[2].flatten().tolist() + [len(Z[c]) - 1.0])
            np.save(os.path.join(o, f"{n}_z.npy"), out)
print({t: len(v) for t, v in LIB.items()}, "K =", K, "valid pairs:", sum(len(v) for v in pairs.values()))
