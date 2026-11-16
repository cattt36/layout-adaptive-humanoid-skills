"""Cross-trial z augmentation (user decision: pilot of the "plan + execute" lower level; exploratory).
Diagnosis: in the student training set z and the action clock always agree (tempo copies re-encode z from the
time-scaled motion), so the students follow the rhythm of z and fail when z comes from another trial (library z on a
new layout; own z under a planned clock: 76 -> 39-55 %).  Here every training clip (fold training trials and their tempo
copies) gets ONE extra copy "<clip>_xz" with the same reference motion / geometry / clock (symlinks; the teacher tracks
it unchanged) but the z_skill of ANOTHER training trial of the same action: one of the 3 geometrically nearest
(layout distance as tools/make_z_library.py; never the same trial or its tempo copies; seeded random choice), linearly
resampled to the clip length -- exactly what the deployment chain supplies (library z + clock from elsewhere).  Within
each pool (originals / "_ts" copies) the copies take half of the sampling mass (equal clip weights), so 50 % of the
episodes see a mismatched z and must take timing / footholds from the obstacle features.
Writes data/aug_xz_f<k>/<geometry dir>/<action>, data/z_tables_cv_f<k>_xz/Z*, clip_split_cv/fold<k>/clips_train_xz.txt.
Mode "w" (argv[2] = w; names *_xzw): the library z is time-warped piecewise linearly so that its stake stance windows
coincide with the clip's own (ThreeStepPoleRun / FiveStepPoleRun; CrouchedTraverse linear) -- z and clock agree as in deployment with the planned clock
(tools/make_z_plan.py plan3), only the trial (style / layout) differs.
Round 2 (user choice B; the linear mode cost ~6 points with the clip's own z):
  25  (names *_xz25 data): per clip one linear cross-z copy + two own-z duplicates ("_d1", "_d2") -> 25 % mismatched
  mix (names *_xzmix):     per clip one linear copy ("_xz") + one stance-aligned copy ("_xzw") + one own-z duplicate
                           ("_d1") -> 50 % mismatched, half linear / half aligned
  python make_xz.py <fold> [w|25|mix]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
X = {"w": "xzw", "25": "xz25", "mix": "xzmix"}.get(sys.argv[2] if len(sys.argv) > 2 else "", "xz")
# copies per clip: (suffix, kind) with kind lin = library z linear, warp = library z stance-aligned, own = own z
RECIPE = {"xz": [("xz", "lin")], "xzw": [("xzw", "warp")], "xz25": [("xz", "lin"), ("d1", "own"), ("d2", "own")],
          "xzmix": [("xz", "lin"), ("xzw", "warp"), ("d1", "own")]}[X]
REL = {"ThreeStepPoleRun": "ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": "ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": "ref_bridge_video_lift/CrouchedTraverse"}
SRC, DST = f"{H}/aug_ts_f{F}", f"{H}/aug_{X}_f{F}"
ZS, ZD = f"{H}/z_tables_cv_f{F}_aug", f"{H}/z_tables_cv_f{F}_{X}"
allow = [ln.strip() for ln in open(f"{H}/clip_split_cv/fold{F}/clips_train_aug.txt") if ln.strip()]
base = lambda c: c.split("_ts")[0]  # noqa: E731
trials = sorted({base(c) for c in allow})
rng = np.random.default_rng()


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def layout(name):
    t = name.split("_")[0]
    d = os.path.join(SRC, REL[t])
    ref = np.load(os.path.join(d, f"{name}_motion.npz"))
    p0, yaw0 = ref["body_pos_w"][0, 0], yaw_of(ref["body_quat_w"][0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rot = lambda xy: np.array([c * (xy[0] - p0[0]) - s * (xy[1] - p0[1]), s * (xy[0] - p0[0]) + c * (xy[1] - p0[1])])  # noqa: E731
    if t == "CrouchedTraverse":
        parts = json.load(open(os.path.join(d, f"{name}_bridge.json")))["parts"]
        pts = [rot(parts["plank"]["center"][:2])] + [rot(parts[p]["center"][:2]) for p in sorted(parts) if p.startswith("post")]
        under = parts["plank"]["center"][2] - parts["plank"]["half"][2]
        return dict(pts=np.array(pts), under=under, yaw=parts["plank"]["yaw"] - yaw0)
    j = json.load(open(os.path.join(d, f"{name}_stakes.json")))
    return dict(pts=np.array([rot(p[:2]) for p in j["stake_xyz_50hz"]]))


def dist(a, b, task):
    if len(a["pts"]) != len(b["pts"]):
        return 1e9
    d = float(np.linalg.norm(a["pts"] - b["pts"], axis=1).mean())
    if task == "CrouchedTraverse":
        d += 3.0 * abs(a["under"] - b["under"]) + 0.5 * abs(np.angle(np.exp(1j * (a["yaw"] - b["yaw"]))))
    return d


L = {n: layout(n) for n in trials}
near = {}
for n in trials:
    t = n.split("_")[0]
    d = {m: dist(L[n], L[m], t) for m in trials if m != n and m.split("_")[0] == t}
    near[n] = [m for m in sorted(d, key=d.get) if d[m] < 1e8][:3]


def windows(name):
    t = name.split("_")[0]
    f = os.path.join(SRC, REL[t], f"{name}_stakes.json")
    if t == "CrouchedTraverse" or not os.path.exists(f):
        return None
    j = json.load(open(f))
    return [x * 50.0 / j["fps"] for s in j["stakes"] for x in s["stance"]]


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


for t, rel in REL.items():
    os.makedirs(f"{DST}/{rel}", exist_ok=True)
    for f in os.listdir(f"{SRC}/{rel}"):
        link(f"{SRC}/{rel}/{f}", f"{DST}/{rel}/{f}")
choice, new = {}, []
for c in allow:
    t = c.split("_")[0]
    if not near[base(c)]:
        continue
    src = near[base(c)][int(rng.integers(len(near[base(c)])))]
    choice[c] = src
    for sfx, _ in RECIPE:
        for ext in ("_motion.npz", "_stakes.json", "_bridge.json"):
            f = f"{SRC}/{REL[t]}/{c}{ext}"
            if os.path.exists(f):
                link(f, f"{DST}/{REL[t]}/{c}_{sfx}{ext}")
        new.append(f"{c}_{sfx}")
nwarp = 0
for P in ["Z0", "Z1", "Z2", "Z3"]:
    os.makedirs(f"{ZD}/{P}", exist_ok=True)
    for f in os.listdir(f"{ZS}/{P}"):
        link(f"{ZS}/{P}/{f}", f"{ZD}/{P}/{f}")
    for c, src in choice.items():
        for sfx, kind in RECIPE:
            if kind == "own":
                link(f"{ZS}/{P}/{c}_z.npy", f"{ZD}/{P}/{c}_{sfx}_z.npy")
                continue
            z, zs = np.load(f"{ZS}/{P}/{c}_z.npy"), np.load(f"{ZS}/{P}/{src}_z.npy")
            ad, asrc = [0.0, len(z) - 1.0], [0.0, len(zs) - 1.0]
            wd, ws = (windows(c), windows(src)) if kind == "warp" else (None, None)
            if wd and ws and len(wd) == len(ws):
                a2, s2 = [0.0] + wd + [len(z) - 1.0], [0.0] + ws + [len(zs) - 1.0]
                if np.all(np.diff(a2) > 0) and np.all(np.diff(s2) > 0):
                    ad, asrc = a2, s2
                    nwarp += P == "Z0"
            tt = np.clip(np.interp(np.arange(len(z)), ad, asrc), 0, len(zs) - 1)
            out = np.stack([np.interp(tt, np.arange(len(zs)), zs[:, k]) for k in range(zs.shape[1])], 1).astype(z.dtype)
            np.save(f"{ZD}/{P}/{c}_{sfx}_z.npy", out)
if any(k == "warp" for _, k in RECIPE):
    print(f"stance-window warps: {nwarp} of {len(choice)} copies (the rest linear: CrouchedTraverse / mismatched windows)")
open(f"{H}/clip_split_cv/fold{F}/clips_train_{X}.txt", "w").write("\n".join(sorted(allow + new)) + "\n")
json.dump(choice, open(f"{ZD}/choice.json", "w"), indent=1)
print(f"fold {F}: {len(allow)} training clips (incl. tempo copies) + {len(new)} cross-z copies; "
      f"trials {len(trials)}; median nearest distance "
      f"{np.median([dist(L[n], L[near[n][0]], n.split('_')[0]) for n in trials if near[n]]):.2f} m")
bad = sum(1 for c in new for P in ["Z0", "Z3"]
          if np.load(f"{DST}/{REL[c.split('_')[0]]}/{c}_motion.npz")["joint_pos"].shape[0] != np.load(f"{ZD}/{P}/{c}_z.npy").shape[0])
print("consistency problems:", bad)
