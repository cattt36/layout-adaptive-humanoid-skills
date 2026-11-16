"""Skill library on a new layout (user request; exploratory): z tables in which every clip uses the z_skill
of a LIBRARY clip (fold training clips, same action, never the clip itself), for evaluating the EXISTING students on
the held-out layouts.  Writes data/z_lib_f<k>/<variant>/Z*/ and data/z_lib_f<k>/choice.json.
Layout descriptor in the clip's start frame (origin = pelvis at frame 0, x = pelvis heading; as deploy/layout_sheet.py):
  ThreeStepPoleRun/2 : stake centres (x, y), in stance order; distance = mean distance of corresponding stakes (m)
  CrouchedTraverse   : plank centre (x, y) and posts (x, y): mean point distance, + 3 x |plank-underside height difference|
                  + 0.5 x |plank yaw difference (rad)|
Variants (the same choice for every level):
  random  : a random library clip of the same action (seed 2026)                          -> lower reference
  geo     : the geometrically nearest library clip, linearly resampled to the clip length
  geowarp : the same clip, time-warped piecewise linearly so that its stake stance windows (start / end) coincide with
            the clip's own (ThreeStepPoleRun / FiveStepPoleRun; CrouchedTraverse = linear)
The clip's own z is the upper reference (original tables).  NOTE: the action clock (duration, stake stance times used
by the obstacle features and the success check) is still that of the held-out trial; only z comes from the library.
  python make_z_library.py <fold>"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SRC = f"{H}/data/z_tables_cv_f{F}"
DST = f"{H}/data/z_lib_f{F}"
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
split = json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))
train = set(split["train"])


def yaw_of(q):  # wxyz
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def layout(name):
    t = name.split("_")[0]
    ref = np.load(os.path.join(GEO[t], f"{name}_motion.npz"))
    p0, yaw0 = ref["body_pos_w"][0, 0], yaw_of(ref["body_quat_w"][0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rot = lambda xy: np.array([c * (xy[0] - p0[0]) - s * (xy[1] - p0[1]), s * (xy[0] - p0[0]) + c * (xy[1] - p0[1])])  # noqa: E731
    T = len(ref["body_pos_w"])
    if t == "CrouchedTraverse":
        j = json.load(open(os.path.join(GEO[t], f"{name}_bridge.json")))
        parts = j["parts"]
        pts = [rot(parts["plank"]["center"][:2])] + [rot(parts[p]["center"][:2]) for p in sorted(parts) if p.startswith("post")]
        under = parts["plank"]["center"][2] - parts["plank"]["half"][2]  # height of the plank underside above the floor
        return dict(T=T, pts=np.array(pts), under=under, yaw=parts["plank"]["yaw"] - yaw0, win=[])
    j = json.load(open(os.path.join(GEO[t], f"{name}_stakes.json")))
    pts = np.array([rot(p[:2]) for p in j["stake_xyz_50hz"]])
    win = [(s_["stance"][0] * 50.0 / j["fps"], s_["stance"][1] * 50.0 / j["fps"]) for s_ in j["stakes"]]
    return dict(T=T, pts=pts, win=win)


def dist(a, b, task):
    if len(a["pts"]) != len(b["pts"]):
        return 1e9
    d = float(np.linalg.norm(a["pts"] - b["pts"], axis=1).mean())
    if task == "CrouchedTraverse":
        d += 3.0 * abs(a["under"] - b["under"]) + 0.5 * abs(np.angle(np.exp(1j * (a["yaw"] - b["yaw"]))))
    return d


def warp(z, T, anchors_dst, anchors_src):
    """Resample z (library clip) to length T: destination frame t -> source frame via piecewise-linear anchors."""
    src_t = np.interp(np.arange(T), anchors_dst, anchors_src)
    src_t = np.clip(src_t, 0, len(z) - 1)
    return np.stack([np.interp(src_t, np.arange(len(z)), z[:, k]) for k in range(z.shape[1])], 1).astype(z.dtype)


allnames = sorted(f[:-6] for f in os.listdir(os.path.join(SRC, "Z0")) if f.endswith("_z.npy"))
# only clips with simulation geometry are ever loaded by the evaluation; the others keep their own z
names = [n for n in allnames if os.path.exists(os.path.join(GEO[n.split("_")[0]], f"{n}_motion.npz"))]
L = {n: layout(n) for n in names}
rng = np.random.default_rng(2026)
choice = {}
for n in names:
    t = n.split("_")[0]
    lib = [m for m in names if m in train and m != n and m.split("_")[0] == t]
    d = {m: dist(L[n], L[m], t) for m in lib}
    geo = min(d, key=d.get)
    rnd = lib[int(rng.integers(len(lib)))]
    choice[n] = dict(task=t, split=next((k for k in ("train", "val", "test") if n in split[k]), "?"), geo=geo,
                     geo_dist_m=round(float(d[geo]), 3), random=rnd, random_dist_m=round(float(d[rnd]), 3),
                     median_lib_dist_m=round(float(np.median(list(d.values()))), 3))
os.makedirs(DST, exist_ok=True)
# HNU_LEVELS=ZR builds only the given levels (fair "explicit reference" baseline) and leaves choice.json
LEVELS = os.environ.get("HNU_LEVELS", "Z0,Z1,Z2,Z3,Z4").split(",")
if "HNU_LEVELS" not in os.environ or not os.path.exists(os.path.join(DST, "choice.json")):
    json.dump(choice, open(os.path.join(DST, "choice.json"), "w"), indent=1)
for P in LEVELS:
    d = os.path.join(SRC, P)
    Z = {n: np.load(os.path.join(d, f"{n}_z.npy")) for n in allnames}
    for v in ("random", "geo", "geowarp"):
        o = os.path.join(DST, v, P)
        os.makedirs(o, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith("_z.npy") and os.path.isfile(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), o)
        for n in allnames:
            if n not in choice:  # no simulation geometry: never evaluated, own z kept
                np.save(os.path.join(o, f"{n}_z.npy"), Z[n])
                continue
            T = len(Z[n])
            src = choice[n]["random" if v == "random" else "geo"]
            zs = Z[src]
            if v == "geowarp" and L[n]["win"] and len(L[n]["win"]) == len(L[src]["win"]):
                ad = [0.0] + [x for w in L[n]["win"] for x in w] + [T - 1.0]
                asrc = [0.0] + [x for w in L[src]["win"] for x in w] + [len(zs) - 1.0]
                ok = np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)
                out = warp(zs, T, ad, asrc) if ok else warp(zs, T, [0, T - 1], [0, len(zs) - 1])
            else:
                out = warp(zs, T, [0, T - 1], [0, len(zs) - 1])
            assert out.shape == Z[n].shape
            np.save(os.path.join(o, f"{n}_z.npy"), out)
te = {n: c for n, c in choice.items() if c["split"] == "test"}
for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    v = [c for c in te.values() if c["task"] == t]
    print(t, f"test clips {len(v)}: geo dist median {np.median([c['geo_dist_m'] for c in v]):.3f} m, random "
          f"{np.median([c['random_dist_m'] for c in v]):.3f} m, median over library {np.median([c['median_lib_dist_m'] for c in v]):.3f} m")
