"""Student-training set with tempo augmentation.  TRAIN trials replayed at other speeds (tools/make_timescale.py);
z of the copies re-encoded by the trained v5t latent encoder from the time-scaled motion (export_z_tables_ts.py), so
their tempo content is consistent.  Speeds per action are limited to where the teacher bank stays reliable
(runs/ts_check, train trials, nominal): ThreeStepPoleRun / FiveStepPoleRun >= 94 % in 0.9-1.15x; CrouchedTraverse 91 / 91 % at 0.95 / 1.05x
(92 % at 1.0x) but 79 / 83 % at 0.9 / 1.1x.
  python make_aug_ts.py k   -> v6g10k set: ThreeStepPoleRun / FiveStepPoleRun at 0.90 / 0.95 / 1.05 / 1.10x, no CrouchedTraverse copies
  python make_aug_ts.py m   -> v6g10m set: same + CrouchedTraverse at 0.95 / 1.05x
  python make_aug_ts.py n   -> v6g10n set: as m, built from the split-v2 training trials (clips_train_v2.txt)
Writes (symlinks, originals untouched; suffix "" for k, "_m" for m):
  data/aug_ts<sfx>/<geometry dir>/<action>   originals + <clip>_tsXXX
  data/z_tables_v5t_aug<sfx>/Z0..Z4          originals + <clip>_tsXXX
  data/clip_split_trial/clips_train_aug<sfx>.txt   train trials + their copies
Evaluation keeps using the original directories / z tables (no copies)."""
import glob
import os
import sys

H = "/root/gpufree-data/hnu_v2/data"
SET = sys.argv[1] if len(sys.argv) > 1 else "k"
SFX = "" if SET == "k" else f"_{SET}"
DIRS = {"ref_stakes_s93_h10/ThreeStepPoleRun": ["090", "095", "105", "110"],
        "ref_stakes_s83_h10/FiveStepPoleRun": ["090", "095", "105", "110"]}
CV = SET.startswith("f")  # cross-validation fold f2 / f3 / f4: the fold's own train trials, latents and z tables
if SET in ("m", "n") or CV:
    DIRS["ref_bridge_video_lift/CrouchedTraverse"] = ["095", "105"]
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
# set n (v6g10n): split v2 training trials (data-quality exclusions ThreeStepPoleRun_022 / ThreeStepPoleRun_025 / ThreeStepPoleRun_029 removed)
if CV:
    TRAIN_FILE = f"{H}/clip_split_cv/fold{SET[1:]}/clips_train.txt"
    ALLOW_OUT = f"{H}/clip_split_cv/fold{SET[1:]}/clips_train_aug.txt"
    ZBASE, ZENC = f"{H}/z_tables_cv_{SET}", f"z_enc_{SET}"
else:
    TRAIN_FILE = f"{H}/clip_split_trial/" + ("clips_train_v2.txt" if SET == "n" else "clips_train.txt")
    ALLOW_OUT = f"{H}/clip_split_trial/clips_train_aug{SFX}.txt"
    ZBASE, ZENC = f"{H}/z_tables_v5t", "z_enc"
train = [ln.strip() for ln in open(TRAIN_FILE) if ln.strip()]


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


copies = []
for rel, speeds in DIRS.items():
    dst = f"{H}/aug_ts{SFX}/{rel}"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{H}/{rel}/*"):
        link(f, f"{dst}/{os.path.basename(f)}")
    for sp in speeds:
        for f in glob.glob(f"{H}/ts/ts{sp}/{rel}/*_motion.npz"):
            name = os.path.basename(f).replace("_motion.npz", "")
            if name not in train:
                continue
            new = f"{name}_ts{sp}"
            link(f, f"{dst}/{new}_motion.npz")
            for ext in ("_stakes.json", "_bridge.json"):
                if os.path.exists(f.replace("_motion.npz", ext)):
                    link(f.replace("_motion.npz", ext), f"{dst}/{new}{ext}")
            copies.append((name, sp, new))
ZOUT = f"{H}/z_tables_v5t_aug{SFX}" if not CV else f"{H}/z_tables_cv_{SET}_aug"
for P in LV:
    dst = f"{ZOUT}/{P}"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{ZBASE}/{P}/*"):
        link(f, f"{dst}/{os.path.basename(f)}")
    for name, sp, new in copies:
        src = f"{H}/ts/ts{sp}/{ZENC}/{P}/{name}_z.npy"
        assert os.path.exists(src), src
        link(src, f"{dst}/{new}_z.npy")
allow = sorted(train + [c[2] for c in copies])
open(ALLOW_OUT, "w").write("\n".join(allow) + "\n")
print(f"set {SET}: {len(train)} train trials + {len(copies)} tempo copies " + ", ".join(
    f"{sum(c[0].startswith(t) for c in copies)} {t}" for t in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")))
# consistency: every allowed clip has a motion file and, per level, a z table of the same length
import numpy as np  # noqa: E402
mdir = {"ThreeStepPoleRun": f"{H}/aug_ts{SFX}/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/aug_ts{SFX}/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/aug_ts{SFX}/ref_bridge_video_lift/CrouchedTraverse"}
bad = 0
for P in LV:
    for c in allow:
        m, z = f"{mdir[c.split('_')[0]]}/{c}_motion.npz", f"{ZOUT}/{P}/{c}_z.npy"
        if not (os.path.exists(m) and os.path.exists(z)) or np.load(m)["joint_pos"].shape[0] != np.load(z).shape[0]:
            bad += 1
print(f"consistency check: {len(allow)} clips x {len(LV)} levels, problems: {bad}")
