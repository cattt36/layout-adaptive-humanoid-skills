"""8.3.2 deployment conditions, new layout perturbations (user decision; conservative ranges, reported only
if the result is acceptable).  For each setting a geometry directory and planner tables are written such that the
deployed chain is re-run on the PERTURBED layout exactly as on the robot: library clip = geometrically nearest fold
training clip for the perturbed layout, stance windows planned from the perturbed stake positions (plan4 rules of
tools/make_z_plan.py: first onset from the training-clip regression, swing segments scaled by the distance ratio,
stance durations of the library clip), library z warped to the planned windows.
  spacing<s>   ThreeStepPoleRun / FiveStepPoleRun: distances from the first stake scaled by s (first stake fixed)           s = 0.90 0.95 1.05 1.10
  start<d>     ThreeStepPoleRun / FiveStepPoleRun: every stake moved by d metres along the start -> first-stake direction     d = -0.10 -0.05 +0.05 +0.10
  post<g>      CrouchedTraverse: the posts on either side of the crossing point moved apart (+) / together (-) by g/2 each
               (gap changes by g; plan unchanged: the beam entry does not depend on the posts)    g = -0.05 -0.03 +0.03 +0.05
Base = plan4 (geom_plan4_f<F>, z_lib_f<F>/plan4).  Writes data/geom_pert_f<F>/<setting>/{ref_stakes_s93_h10/ThreeStepPoleRun,
ref_stakes_s83_h10/FiveStepPoleRun,ref_bridge_video_lift/CrouchedTraverse} and data/z_pert_f<F>/<setting>/Z0..Z3.
  python make_layout_perturb.py <fold>"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
GEO = {"ThreeStepPoleRun": f"{H}/ref_stakes_s93_h10", "FiveStepPoleRun": f"{H}/ref_stakes_s83_h10"}
BR = f"{H}/ref_bridge_video_lift"
GP = f"{H}/geom_plan4_f{F}"
ZP = f"{H}/z_lib_f{F}/plan4"
ZS = f"{H}/z_tables_cv_f{F}"
LV = ["Z0", "Z1", "Z2", "Z3"]
split = json.load(open(f"{H}/clip_split_cv/fold{F}/split.json"))
train = set(split["train"])
SETTINGS = [("spacing", s) for s in (0.90, 0.95, 1.05, 1.10)] + [("start", d) for d in (-0.10, -0.05, 0.05, 0.10)] + \
           [("post", g) for g in (-0.05, -0.03, 0.03, 0.05)]


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def stakes(name):
    t = name.split("_")[0]
    j = json.load(open(os.path.join(GEO[t], t, f"{name}_stakes.json")))
    ref = np.load(os.path.join(GEO[t], t, f"{name}_motion.npz"))
    pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
    win = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
    return dict(j=j, p0=ref["body_pos_w"][0, 0, :2], yaw0=yaw_of(ref["body_quat_w"][0, 0]), pts=pts, win=win, T=len(ref["body_pos_w"]))


def rel(p, p0, yaw0):
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return (p - p0) @ np.array([[c, -s], [s, c]]).T


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def warp(z, T, ad, asrc):
    if not (np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)):
        ad, asrc = [0.0, T - 1.0], [0.0, len(z) - 1.0]
    tt = np.clip(np.interp(np.arange(T), ad, asrc), 0, len(z) - 1)
    return np.stack([np.interp(tt, np.arange(len(z)), z[:, k]) for k in range(z.shape[1])], 1).astype(z.dtype)


names = {t: sorted(f[:-len("_stakes.json")] for f in os.listdir(os.path.join(GEO[t], t)) if f.endswith("_stakes.json")) for t in GEO}
ST = {n: stakes(n) for t in GEO for n in names[t]}
REG = {t: np.polyfit([np.linalg.norm(ST[n]["pts"][0] - ST[n]["p0"]) for n in names[t] if n in train],
                     [ST[n]["win"][0, 0] for n in names[t] if n in train], 1) for t in GEO}
ZT = {P: {} for P in LV}


def ztab(P, n):
    if n not in ZT[P]:
        ZT[P][n] = np.load(os.path.join(ZS, P, f"{n}_z.npy"))
    return ZT[P][n]


def plan(n, pts):
    """library choice + planned windows for clip n with perturbed stake positions pts (plan4 rules)."""
    t = n.split("_")[0]
    a = ST[n]
    rn = rel(pts, a["p0"], a["yaw0"])
    lib = [m for m in names[t] if m in train and m != n and len(ST[m]["pts"]) == len(pts)]
    c = min(lib, key=lambda m: float(np.linalg.norm(rn - rel(ST[m]["pts"], ST[m]["p0"], ST[m]["yaw0"]), axis=1).mean()))
    b = ST[c]
    seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
    r = np.clip(seg(a["p0"], pts) / np.maximum(seg(b["p0"], b["pts"]), 1e-3), 0.6, 1.6)
    ws = b["win"]
    gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]
    stance = ws[:, 1] - ws[:, 0]
    out, tt = [], 0.0
    for k in range(len(ws)):
        s0 = float(np.polyval(REG[t], np.linalg.norm(pts[0] - a["p0"]))) if k == 0 else tt + gaps[k] * r[k]
        out.append((s0, s0 + stance[k]))
        tt = s0 + stance[k]
    out = np.array(out)
    if out[-1, 1] > a["T"] - 11:
        out *= (a["T"] - 11) / out[-1, 1]
    return c, out


def crossing(name, parts):
    """position along the plank axis where the reference pelvis passes under the plank axis line."""
    m = np.load(os.path.join(BR, "CrouchedTraverse", f"{name}_motion.npz"))
    c, yaw = np.array(parts["plank"]["center"][:2]), parts["plank"]["yaw"]
    d, nn = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rp = m["body_pos_w"][:, 0, :2] - c
    side = rp @ nn
    i = int(np.argmin(np.abs(side)))
    return float(rp[i] @ d), c, d


# self-check: on the UNperturbed layout the re-planning must reproduce plan4 (library choice and stance windows)
dev, nchk = [], 0
for t in GEO:
    for n in names[t]:
        f = os.path.join(GP, os.path.basename(GEO[t]), t, f"{n}_stakes.json")
        j = json.load(open(f))
        if "planned_from" not in j:
            continue
        c, wn = plan(n, ST[n]["pts"])
        wp = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
        nchk += 1
        dev.append(float(np.abs(wn - wp).max()) if c == j["planned_from"] else 1e9)
print(f"self-check vs plan4: {nchk} clips, library choice identical {sum(d < 1e8 for d in dev)}, "
      f"max window deviation {max([d for d in dev if d < 1e8] or [0]):.2f} frames (rounding < 1.5 expected)")

summary = {}
for kind, v in SETTINGS:
    name = f"{kind}{v:+.2f}" if kind != "spacing" else f"{kind}{v:.2f}"
    G, Z = f"{H}/geom_pert_f{F}/{name}", f"{H}/z_pert_f{F}/{name}"
    for t in list(GEO) + ["CrouchedTraverse"]:
        sub = os.path.join(os.path.basename(GEO[t]), t) if t in GEO else os.path.join("ref_bridge_video_lift", t)
        src = os.path.join(GP, os.path.basename(GEO[t]), t) if t in GEO else os.path.join(BR, t)
        os.makedirs(os.path.join(G, sub), exist_ok=True)
        for f in os.listdir(src):
            link(os.path.join(src, f), os.path.join(G, sub, f))
    for P in LV:
        os.makedirs(os.path.join(Z, P), exist_ok=True)
        for f in os.listdir(os.path.join(ZP, P)):
            link(os.path.join(ZP, P, f), os.path.join(Z, P, f))
    moved = []
    if kind in ("spacing", "start"):
        for t in GEO:
            for n in names[t]:
                a = ST[n]
                pts = a["pts"].copy()
                if kind == "spacing":
                    pts[1:] = pts[0] + v * (pts[1:] - pts[0])
                else:
                    u = (pts[0] - a["p0"]) / max(np.linalg.norm(pts[0] - a["p0"]), 1e-6)
                    pts = pts + v * u
                c, wn = plan(n, pts)
                j = json.loads(json.dumps(a["j"]))
                for k, (s_, p_) in enumerate(zip(j["stakes"], pts)):
                    s_["x"], s_["y"] = float(p_[0]), float(p_[1])
                    s_["stance"] = [int(round(wn[k, 0] * j["fps"] / 50.0)), int(round(wn[k, 1] * j["fps"] / 50.0))]
                    j["stake_xyz_50hz"][k][0], j["stake_xyz_50hz"][k][1] = float(p_[0]), float(p_[1])
                j["planned_from"], j["perturbation"] = c, name
                dst = os.path.join(G, os.path.basename(GEO[t]), t, f"{n}_stakes.json")
                os.remove(dst)
                json.dump(j, open(dst, "w"))
                ad = [0.0] + wn.flatten().tolist() + [a["T"] - 1.0]
                asrc = [0.0] + ST[c]["win"].flatten().tolist() + [ST[c]["T"] - 1.0]
                for P in LV:
                    zn = ztab(P, n)
                    out = warp(ztab(P, c), len(zn), ad, asrc)
                    dz = os.path.join(Z, P, f"{n}_z.npy")
                    os.remove(dz)
                    np.save(dz, out)
                if split.get("test") and n in split["test"]:
                    moved.append(float(np.abs(pts - a["pts"]).max()))
    else:  # posts
        for f in sorted(os.listdir(os.path.join(BR, "CrouchedTraverse"))):
            if not f.endswith("_bridge.json"):
                continue
            n = f[:-len("_bridge.json")]
            bj = json.load(open(os.path.realpath(os.path.join(BR, "CrouchedTraverse", f))))
            parts = bj["parts"]
            x0, c, d = crossing(n, parts)
            for p in parts:
                if p.startswith("post"):
                    xp = float((np.array(parts[p]["center"][:2]) - c) @ d)
                    sgn = 1.0 if xp > x0 else -1.0
                    parts[p]["center"][0] += float(sgn * v / 2 * d[0])
                    parts[p]["center"][1] += float(sgn * v / 2 * d[1])
            bj["perturbation"] = name
            dst = os.path.join(G, "ref_bridge_video_lift", "CrouchedTraverse", f)
            os.remove(dst)
            json.dump(bj, open(dst, "w"))
            if n in split.get("test", []):
                moved.append(abs(v) / 2)
    summary[name] = dict(test_clips=len(moved), max_move_m=round(max(moved), 3) if moved else 0.0)
print(json.dumps(summary, indent=1))
