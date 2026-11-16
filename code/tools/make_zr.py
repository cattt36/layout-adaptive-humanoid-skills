"""Fair external baseline (user decision; the only baseline reported in the paper): an "explicit reference"
student (H2O / OmniH2O / HOVER interface) trained and evaluated EXACTLY like the new z students (same teacher, same
deployable observations, training trials of the fold only, same cross-trial "mix" recipe, same planner), except that
it receives the reference motion itself instead of z_skill.  Implemented as one more latent level "ZR" whose per-frame
table is the reference command of the clip -- joint positions (29) and joint velocities (29), standardised with the
statistics of the fold's training trials -- so every z-table tool works unchanged (the student reads it at the same
future offsets 0..50 steps as z).
Writes, for fold F:
  data/z_tables_cv_f<F>/ZR          original clips (condition A, library / planner tables)
  data/z_tables_cv_f<F>_aug/ZR      + tempo copies (training pool before the cross-trial copies)
  data/z_tables_cv_f<F>_xzmix/ZR    + the cross-trial copies of tools/make_xz.py mix with the SAME source trials
                                    (choice.json of the z levels): "_xz" linear, "_xzw" stance-aligned, "_d1" own
Clips without a motion file (never loaded) get zeros.  Follow-up (same tools as the z levels, restricted to ZR):
  HNU_LEVELS=ZR tools/make_z_library.py F;  HNU_LEVELS=ZR HNU_SKIP_GEOM=1 tools/make_z_plan.py F plan4
  HNU_LEVELS=ZR HNU_SKIP_GEOM=1 tools/make_z_cands_plan.py F   (4090, E2E sweep)
  python make_zr.py <fold> [--no-xz]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
F = int(sys.argv[1])
NO_XZ = "--no-xz" in sys.argv
REL = {"ThreeStepPoleRun": "ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": "ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": "ref_bridge_video_lift/CrouchedTraverse"}
ORIG = {t: f"{H}/{r}" for t, r in REL.items()}
AUGD = {t: f"{H}/aug_ts_f{F}/{r}" for t, r in REL.items()}
FLAT = {t: f"{H}/ref_v1/{t}" for t in REL}
split = json.load(open(f"{H}/clip_split_cv/fold{F}/split.json"))


def motion(name, dirs):
    t = name.split("_")[0]
    for d in dirs:
        f = os.path.join(d[t], f"{name}_motion.npz")
        if os.path.exists(f):
            m = np.load(f)
            return np.concatenate([m["joint_pos"], m["joint_vel"]], 1).astype(np.float32)
    return None


def names_of(d):
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith("_z.npy"))


# normalisation from the fold's training trials (original clips)
tr = [motion(n, [ORIG]) for n in split["train"]]
tr = np.concatenate([x for x in tr if x is not None])
mu, sd = tr.mean(0), tr.std(0) + 1e-3
stats = dict(mean=mu.tolist(), std=sd.tolist(), dims="joint_pos[29] + joint_vel[29]", fold=F)


def table(name, dirs, T):
    x = motion(name, dirs)
    if x is None or len(x) != T:
        return np.zeros((T, len(mu)), np.float32), x is not None
    return ((x - mu) / sd).astype(np.float32), False


def build(src_level_dir, out_dir, dirs):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_level_dir):
        p = os.path.join(src_level_dir, f)
        if not f.endswith("_z.npy") and os.path.isfile(p):
            open(os.path.join(out_dir, f), "wb").write(open(p, "rb").read())
    json.dump(stats, open(os.path.join(out_dir, "zr_norm.json"), "w"))
    n_zero = n_bad = 0
    for n in names_of(src_level_dir):
        T = len(np.load(os.path.join(src_level_dir, f"{n}_z.npy"), mmap_mode="r"))
        z, bad = table(n, dirs, T)
        n_zero += int(not z.any())
        n_bad += int(bad)
        np.save(os.path.join(out_dir, f"{n}_z.npy"), z)
    print(f"{out_dir}: {len(names_of(src_level_dir))} tables, {n_zero} zero (no motion), {n_bad} length mismatches")


build(f"{H}/z_tables_cv_f{F}/Z0", f"{H}/z_tables_cv_f{F}/ZR", [ORIG, FLAT])
build(f"{H}/z_tables_cv_f{F}_aug/Z0", f"{H}/z_tables_cv_f{F}_aug/ZR", [AUGD, ORIG, FLAT])
if not NO_XZ:
    XD, AD = f"{H}/z_tables_cv_f{F}_xzmix", f"{H}/z_tables_cv_f{F}_aug/ZR"
    choice = json.load(open(f"{XD}/choice.json"))
    out = f"{XD}/ZR"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(AD):
        dst = os.path.join(out, f)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(AD, f)), dst)

    def windows(name):
        t = name.split("_")[0]
        f = os.path.join(AUGD[t], f"{name}_stakes.json")
        if t == "CrouchedTraverse" or not os.path.exists(f):
            return None
        j = json.load(open(f))
        return [x * 50.0 / j["fps"] for s in j["stakes"] for x in s["stance"]]

    nw = 0
    for c, src in choice.items():
        z, zs = np.load(f"{AD}/{c}_z.npy"), np.load(f"{AD}/{src}_z.npy")
        for sfx, kind in (("xz", "lin"), ("xzw", "warp"), ("d1", "own")):
            dst = f"{out}/{c}_{sfx}_z.npy"
            if os.path.lexists(dst):
                os.remove(dst)
            if kind == "own":
                os.symlink(os.path.realpath(f"{AD}/{c}_z.npy"), dst)
                continue
            ad, asrc = [0.0, len(z) - 1.0], [0.0, len(zs) - 1.0]
            wd, ws = (windows(c), windows(src)) if kind == "warp" else (None, None)
            if wd and ws and len(wd) == len(ws):
                a2, s2 = [0.0] + wd + [len(z) - 1.0], [0.0] + ws + [len(zs) - 1.0]
                if np.all(np.diff(a2) > 0) and np.all(np.diff(s2) > 0):
                    ad, asrc = a2, s2
                    nw += 1
            tt = np.clip(np.interp(np.arange(len(z)), ad, asrc), 0, len(zs) - 1)
            np.save(dst, np.stack([np.interp(tt, np.arange(len(zs)), zs[:, k]) for k in range(zs.shape[1])], 1).astype(np.float32))
    # consistency with the z level the students use (same clip names and lengths)
    bad = [n for n in names_of(f"{XD}/Z0") if not os.path.exists(f"{out}/{n}_z.npy")
           or len(np.load(f"{out}/{n}_z.npy", mmap_mode="r")) != len(np.load(f"{XD}/Z0/{n}_z.npy", mmap_mode="r"))]
    print(f"{out}: {len(choice)} cross-trial sources, stance-aligned {nw}; missing / length problems vs Z0: {len(bad)}")
