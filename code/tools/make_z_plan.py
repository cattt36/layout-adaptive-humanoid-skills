"""Layout-adapted skill without human timing (user request; exploratory, follows tools/make_z_library.py).
make_z_library's geowarp aligns the library z to the held-out trial's OWN stake stance windows, i.e. to the human timing
of that trial, which a deployed robot does not know.  Here the timing is planned from the layout only:
  library clip = geometrically nearest fold training clip (data/z_lib_f<k>/choice.json, same as geo / geowarp)
  swing segment k (start pelvis -> stake 1, stake k-1 -> stake k): library duration x (new distance / library distance)
                  (same walking speed; ratio clipped to [0.6, 1.6]); stance durations on the stakes = library's
  -> planned stance windows of the new layout; z = library z time-warped piecewise linearly to them ("plan").
The action clock of the environment is replaced as well: data/geom_plan_f<k>/<s93|s83>_h10/<task>/ = symlinks to the
original geometry, with the ThreeStepPoleRun / FiveStepPoleRun <clip>_stakes.json rewritten to the planned stance windows (the obstacle features'
time-to-step and the stake windows of the success check follow the plan).  CrouchedTraverse has no stake clock: linear
resampling to the clip length (= geo).  Remaining use of the held-out trial: clip length (episode end) and start pose.
Writes data/z_lib_f<k>/plan/Z*/ and data/z_lib_f<k>/plan_timing.json (planned vs actual human windows).
Variant "plan2" (argv[2] = plan2): the library clip's initial standing time (pelvis speed < 0.15 m/s from frame 0) is
capped at 0.5 s and only the walking part of the first swing segment is scaled by the distance ratio (a long initial
wait of the library clip, e.g. FiveStepPoleRun_006, otherwise delays the whole plan by ~2 s).
Variant "plan3": the first stance onset is predicted by a per-action linear regression (onset time ~ distance start pelvis
-> first stake) fitted on the fold's TRAINING clips (library clips with slow / curved approaches, e.g. FiveStepPoleRun_006:
1.03 m in 3.8 s, otherwise dominate); later segments as in "plan".
Variant "plan4" (user: the balance beam must be planned too): ThreeStepPoleRun / FiveStepPoleRun as plan3; CrouchedTraverse: the frame at
which the pelvis first gets under the beam (as tools/make_aug_variants.first_under_beam) is predicted by a linear
regression on the distance start pelvis -> nearer end of the plank axis (fold TRAINING clips), and the library z is
warped piecewise linearly so that its own entry frame falls on the prediction (instead of a linear stretch to the
held-out trial's duration).  No environment clock exists for the beam (obstacle features are purely spatial).
  python make_z_plan.py <fold> [plan|plan2|plan3|plan4]"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SRC = f"{H}/data/z_tables_cv_f{F}"
DST = f"{H}/data/z_lib_f{F}"
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10"}
VAR = sys.argv[2] if len(sys.argv) > 2 else "plan"
GPLAN = f"{H}/data/geom_{VAR}_f{F}"
choice = json.load(open(f"{DST}/choice.json"))


def stakes(name):
    t = name.split("_")[0]
    j = json.load(open(os.path.join(GEO[t], t, f"{name}_stakes.json")))
    ref = np.load(os.path.join(GEO[t], t, f"{name}_motion.npz"))
    pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
    win = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
    pel = ref["body_pos_w"][:, 0, :2]
    v = np.linalg.norm(np.diff(pel, axis=0), axis=1) * 50.0
    onset = int(np.argmax(np.convolve(v, np.ones(5) / 5, "same") > 0.15))
    return j, pel[0], pts, win, len(ref["body_pos_w"]), onset


def plan(dst, src):
    _, p0d, pd, _, Td, _ = stakes(dst)
    _, p0s, ps, ws, _, on = stakes(src)
    seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
    r = np.clip(seg(p0d, pd) / np.maximum(seg(p0s, ps), 1e-3), 0.6, 1.6)
    gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]  # swing before each stance (first: from frame 0)
    stance = ws[:, 1] - ws[:, 0]
    wait = 0.0
    if VAR == "plan2" and 0 < on < gaps[0]:
        wait, gaps = min(on, 25.0), gaps.copy()
        gaps[0] -= on
    out, t = [], wait
    for k in range(len(ws)):
        a = t + gaps[k] * r[k]
        if k == 0 and VAR in ("plan3", "plan4"):
            a = float(np.polyval(REG[dst.split("_")[0]], np.linalg.norm(pd[0] - p0d)))
        out.append((a, a + stance[k]))
        t = a + stance[k]
    out = np.array(out)
    if out[-1, 1] > Td - 11:  # does not fit before the clip end: compress uniformly
        s_ = (Td - 11) / out[-1, 1]
        out, wait = out * s_, wait * s_
    return out, ws, (wait, float(on)) if wait > 0 else None


def warp(z, T, anchors_dst, anchors_src):
    src_t = np.clip(np.interp(np.arange(T), anchors_dst, anchors_src), 0, len(z) - 1)
    return np.stack([np.interp(src_t, np.arange(len(z)), z[:, k]) for k in range(z.shape[1])], 1).astype(z.dtype)


BRIDGE = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"


def beam(name):
    """(distance start pelvis -> nearer end of the plank axis, entry frame under the beam, clip length)."""
    m = np.load(os.path.join(BRIDGE, f"{name}_motion.npz"))
    p = json.load(open(os.path.realpath(os.path.join(BRIDGE, f"{name}_bridge.json"))))["parts"]["plank"]
    c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - c[:2]
    inside = (np.abs(rel @ n) < 0.25) & (np.abs(rel @ d) < half[0])
    entry = int(np.where(inside.any(1))[0][0]) if inside.any() else -1
    p0 = m["body_pos_w"][0, 0, :2]
    ends = [c[:2] + d * half[0], c[:2] - d * half[0]]
    return float(min(np.linalg.norm(e - p0) for e in ends)), entry, len(m["body_pos_w"])


PLAN_D = {}
if VAR == "plan4":
    B = {n: beam(n) for n, c in choice.items() if c["task"] == "CrouchedTraverse"}
    xy = [(B[n][0], B[n][1]) for n, c in choice.items() if c["task"] == "CrouchedTraverse" and c["split"] == "train" and B[n][1] > 0]
    RD = np.polyfit(*np.array(xy, float).T, 1)
    res = np.array([y - np.polyval(RD, x) for x, y in xy]) / 50
    print(f"CrouchedTraverse entry regression on {len(xy)} training clips: {RD[1] / 50:.2f} s + {RD[0] / 50:.2f} s/m, residual sd {res.std():.2f} s")
    err = []
    for n, c in choice.items():
        if c["task"] != "CrouchedTraverse" or B[n][1] <= 0 or B[c["geo"]][1] <= 0:
            continue
        pred = float(np.clip(np.polyval(RD, B[n][0]), 10, B[n][2] - 30))
        PLAN_D[n] = (pred, float(B[c["geo"]][1]))
        if c["split"] == "test":
            err.append(abs(pred - B[n][1]) / 50)
    print(f"CrouchedTraverse test clips {len(err)}: planned vs human entry under the beam, mean abs error {np.mean(err):.2f} s (max {np.max(err):.2f})")
REG = {}
if VAR in ("plan3", "plan4"):
    for t in GEO:
        xy = [(np.linalg.norm(s[2][0] - s[1]), s[3][0, 0]) for n, c in choice.items()
              if c["task"] == t and c["split"] == "train" for s in [stakes(n)]]
        REG[t] = np.polyfit(*np.array(xy).T, 1)
        res = np.array([y - np.polyval(REG[t], x) for x, y in xy]) / 50
        print(t, f"onset regression on {len(xy)} training clips: {REG[t][1] / 50:.2f} s + {REG[t][0] / 50:.2f} s/m, "
              f"residual sd {res.std():.2f} s")
timing, plans = {}, {}
for n, c in choice.items():
    if c["task"] not in GEO:
        continue
    _, _, pd, wd, _, _ = stakes(n)
    _, _, ps, _, _, _ = stakes(c["geo"])
    if len(pd) != len(ps):
        continue
    pw, sw, wo = plan(n, c["geo"])
    plans[n] = (pw, sw, wo)
    timing[n] = dict(split=c["split"], lib=c["geo"], planned=np.round(pw / 50, 2).tolist(), human=np.round(wd / 50, 2).tolist(),
                     mean_abs_err_s=round(float(np.abs(pw - wd).mean() / 50), 3))
# HNU_SKIP_GEOM=1 (with HNU_LEVELS=ZR): only z tables of the given levels; geometry / timing untouched
SKIP_GEOM = os.environ.get("HNU_SKIP_GEOM", "0") == "1"
if not SKIP_GEOM:
    json.dump(timing, open(f"{DST}/{VAR}_timing.json", "w"), indent=1)

# environment clock: geometry copies with the planned stance windows
for t, g in ({} if SKIP_GEOM else GEO).items():
    o = os.path.join(GPLAN, os.path.basename(g), t)
    os.makedirs(o, exist_ok=True)
    for f in os.listdir(os.path.join(g, t)):
        dst = os.path.join(o, f)
        if os.path.lexists(dst):
            os.remove(dst)
        n = f[:-len("_stakes.json")] if f.endswith("_stakes.json") else None
        if n in plans:
            j = json.load(open(os.path.join(g, t, f)))
            for s, (a, b) in zip(j["stakes"], plans[n][0]):
                s["stance"] = [int(round(a * j["fps"] / 50.0)), int(round(b * j["fps"] / 50.0))]
            j["planned_from"] = choice[n]["geo"]
            json.dump(j, open(dst, "w"))
        else:
            os.symlink(os.path.realpath(os.path.join(g, t, f)), dst)

# z tables
for P in os.environ.get("HNU_LEVELS", "Z0,Z1,Z2,Z3").split(","):
    d, o = os.path.join(SRC, P), os.path.join(DST, VAR, P)
    os.makedirs(o, exist_ok=True)
    for f in os.listdir(d):
        if not f.endswith("_z.npy") and os.path.isfile(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), o)
    geo = os.path.join(DST, "geo", P)  # CrouchedTraverse / clips without a plan: the geo table (linear resampling)
    for f in os.listdir(geo):
        if f.endswith("_z.npy"):
            n = f[:-6]
            if n in plans:
                z, zs = np.load(os.path.join(d, f)), np.load(os.path.join(d, f"{choice[n]['geo']}_z.npy"))
                T = len(z)
                pw, sw, wo = plans[n]
                ad = [0.0] + ([wo[0]] if wo else []) + pw.flatten().tolist() + [T - 1.0]  # wait end <- library onset
                asrc = [0.0] + ([wo[1]] if wo else []) + sw.flatten().tolist() + [len(zs) - 1.0]
                out = warp(zs, T, ad, asrc) if np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0) else \
                    warp(zs, T, [0, T - 1], [0, len(zs) - 1])
                np.save(os.path.join(o, f), out)
            elif n in PLAN_D:  # beam: library entry frame -> planned entry frame
                z, zs = np.load(os.path.join(d, f)), np.load(os.path.join(d, f"{choice[n]['geo']}_z.npy"))
                pred, le = PLAN_D[n]
                ad, asrc = [0.0, pred, len(z) - 1.0], [0.0, le, len(zs) - 1.0]
                ok = np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)
                np.save(os.path.join(o, f), warp(zs, len(z), ad, asrc) if ok else warp(zs, len(z), [0, len(z) - 1], [0, len(zs) - 1]))
            else:
                shutil.copy(os.path.join(geo, f), o)
te = {n: v for n, v in timing.items() if v["split"] == "test"}
for t in GEO:
    e = [v["mean_abs_err_s"] for n, v in te.items() if n.startswith(t)]
    print(t, f"test clips {len(e)}: planned vs human stance windows, mean abs error {np.mean(e):.2f} s (max {np.max(e):.2f})")
